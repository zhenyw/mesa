/*
 * Copyright © 2013 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/**
 * \file brw_performance_query.c
 *
 * Implementation of the GL_INTEL_performance_query extension.
 *
 * Currently there are two possible counter sources exposed here:
 *
 * On Gen6+ hardware we have numerous 64bit Pipeline Statistics Registers
 * that we can snapshot at the beginning and end of a query.
 *
 * On Gen7.5+ we have Observability Architecture counters which are
 * covered in separate document from the rest of the PRMs.  It is available at:
 * https://01.org/linuxgraphics/documentation/driver-documentation-prms
 * => 2013 Intel Core Processor Family => Observability Performance Counters
 * (This one volume covers Sandybridge, Ivybridge, Baytrail, and Haswell,
 * though notably we currently only support OA counters for Haswell+)
 */

#include <linux/perf_event.h>

#include <limits.h>

#include <asm/unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

#include "main/hash.h"
#include "main/macros.h"
#include "main/mtypes.h"
#include "main/performance_query.h"

#include "util/bitset.h"
#include "util/ralloc.h"

#include "brw_context.h"
#include "brw_defines.h"
#include "intel_batchbuffer.h"

#define FILE_DEBUG_FLAG DEBUG_PERFQUERY

/* Describes how to read one OA counter which might be a raw counter read
 * directly from a counter snapshot or could be a higher level counter derived
 * from one or more raw counters.
 *
 * Raw counters will have set ->report_offset to the snapshot offset and have
 * an accumulator that can consider counter overflow according to the width of
 * that counter.
 *
 * Higher level counters can currently reference up to 3 other counters + use
 * ->config for anything. They don't need an accumulator.
 *
 * The data type that will be written to *value_out by the read function can
 * be determined by ->data_type
 */
struct brw_oa_counter
{
   struct brw_oa_counter *reference0;
   struct brw_oa_counter *reference1;
   struct brw_oa_counter *reference2;
   union {
      int report_offset;
      int config;
   };

   int accumulator_index;
   void (*accumulate)(struct brw_oa_counter *counter,
                      uint32_t *start,
                      uint32_t *end,
                      uint64_t *accumulator);
   GLenum data_type;
   void (*read)(struct brw_oa_counter *counter,
                uint64_t *accumulated,
                void *value_out);
};

/* A counter that will be advertised and reported to applications */
struct brw_perf_query_counter
{
   const char *name;
   const char *desc;
   GLenum type;
   GLenum data_type;
   uint64_t raw_max;
   size_t offset;
   size_t size;

   union {
      struct brw_oa_counter *oa_counter;
      uint32_t pipeline_stat_reg;
   };
};

struct brw_query_builder
{
   struct brw_context *brw;
   struct brw_perf_query *query;
   size_t offset;
   int next_accumulator_index;

   int a_offset;
   int b_offset;
   int c_offset;

   struct brw_oa_counter *gpu_core_clock;
};

/**
 * i965 representation of a performance query object.
 *
 * NB: We want to keep this structure relatively lean considering that
 * applications may expect to allocate enough objects to be able to
 * query around all draw calls in a frame.
 */
struct brw_perf_query_object
{
   /** The base class. */
   struct gl_perf_query_object base;

   const struct brw_perf_query *query;

   /* See query->kind to know which state below is in use... */
   union {
      struct {

         /**
          * BO containing OA counter snapshots at query Begin/End time.
          */
         drm_intel_bo *bo;
         int current_report_id;

         /**
          * We collect periodic counter snapshots via perf so we can account
          * for counter overflow and this is a pointer into the circular
          * perf buffer for collecting snapshots that lie within the begin-end
          * bounds of this query.
          */
         unsigned int perf_tail;

         /**
          * Storage the final accumulated OA counters.
          */
         uint64_t accumulator[MAX_RAW_OA_COUNTERS];

         /**
          * false while in the unresolved_elements list, and set to true when
          * the final, end MI_RPC snapshot has been accumulated.
          */
         bool results_accumulated;

      } oa;

      struct {
         /**
          * BO containing starting and ending snapshots for the
          * statistics counters.
          */
         drm_intel_bo *bo;

         /**
          * Storage for final pipeline statistics counter results.
          */
         uint64_t *results;

      } pipeline_stats;
   };
};

/* Samples read from the perf circular buffer */
struct oa_perf_sample {
   struct perf_event_header header;
   uint32_t raw_size;
   uint8_t raw_data[];
};
#define MAX_OA_PERF_SAMPLE_SIZE (8 +   /* perf_event_header */       \
                                 4 +   /* raw_size */                \
                                 256 + /* raw OA counter snapshot */ \
                                 4)    /* alignment padding */

#define TAKEN(HEAD, TAIL, POT_SIZE)	(((HEAD) - (TAIL)) & (POT_SIZE - 1))

/* Note: this will equate to 0 when the buffer is exactly full... */
#define REMAINING(HEAD, TAIL, POT_SIZE) (POT_SIZE - TAKEN (HEAD, TAIL, POT_SIZE))

#if defined(__i386__)
#define rmb()           __asm__ volatile("lock; addl $0,0(%%esp)" ::: "memory")
#define mb()            __asm__ volatile("lock; addl $0,0(%%esp)" ::: "memory")
#endif

#if defined(__x86_64__)
#define rmb()           __asm__ volatile("lfence" ::: "memory")
#define mb()            __asm__ volatile("mfence" ::: "memory")
#endif

/* TODO: consider using <stdatomic.h> something like:
 *
 * #define rmb() atomic_thread_fence(memory_order_seq_consume)
 * #define mb() atomic_thread_fence(memory_order_seq_cst)
 */

/* Allow building for a more recent kernel than the system headers
 * correspond too... */
#ifndef PERF_EVENT_IOC_FLUSH
#include <linux/ioctl.h>
#define PERF_EVENT_IOC_FLUSH                 _IO ('$', 8)
#endif


/* attr.config */

#define I915_PERF_OA_CTX_ID_MASK	    0xffffffff
#define I915_PERF_OA_SINGLE_CONTEXT_ENABLE  (1ULL << 32)

#define I915_PERF_OA_FORMAT_SHIFT	    33
#define I915_PERF_OA_FORMAT_MASK	    (0x7ULL << 33)
#define I915_PERF_OA_FORMAT_A13_HSW	    (0ULL << 33)
#define I915_PERF_OA_FORMAT_A29_HSW	    (1ULL << 33)
#define I915_PERF_OA_FORMAT_A13_B8_C8_HSW   (2ULL << 33)
#define I915_PERF_OA_FORMAT_B4_C8_HSW	    (4ULL << 33)
#define I915_PERF_OA_FORMAT_A45_B8_C8_HSW   (5ULL << 33)
#define I915_PERF_OA_FORMAT_B4_C8_A16_HSW   (6ULL << 33)
#define I915_PERF_OA_FORMAT_C4_B8_HSW	    (7ULL << 33)

#define I915_PERF_OA_TIMER_EXPONENT_SHIFT   36
#define I915_PERF_OA_TIMER_EXPONENT_MASK    (0x3fULL << 36)

#define I915_PERF_OA_PROFILE_SHIFT          42
#define I915_PERF_OA_PROFILE_MASK           (0x3fULL << 42)
#define I915_PERF_OA_PROFILE_3D             1


/** Downcasting convenience macro. */
static inline struct brw_perf_query_object *
brw_perf_query(struct gl_perf_query_object *o)
{
   return (struct brw_perf_query_object *) o;
}

#define SECOND_SNAPSHOT_OFFSET_IN_BYTES 2048

static inline size_t
pot_align(size_t base, int pot_alignment)
{
    return (base + pot_alignment - 1) & ~(pot_alignment - 1);
}

/******************************************************************************/

static GLboolean brw_is_perf_query_ready(struct gl_context *,
					 struct gl_perf_query_object *);

static void
dump_perf_query_callback(GLuint id, void *query_void, void *brw_void)
{
   struct gl_context *ctx = brw_void;
   struct gl_perf_query_object *o = query_void;
   struct brw_perf_query_object *obj = query_void;

   switch(obj->query->kind) {
   case OA_COUNTERS:
      DBG("%4d: %-6s %-8s BO: %-4s OA data: %-10s %-15s\n",
          id,
          o->Used ? "Dirty," : "New,",
          o->Active ? "Active," : (o->Ready ? "Ready," : "Pending,"),
          obj->oa.bo ? "yes," : "no,",
          brw_is_perf_query_ready(ctx, o) ? "ready," : "not ready,",
          obj->oa.results_accumulated ? "accumulated" : "not accumulated");
      break;
   case PIPELINE_STATS:
      DBG("%4d: %-6s %-8s BO: %-4s\n",
          id,
          o->Used ? "Dirty," : "New,",
          o->Active ? "Active," : (o->Ready ? "Ready," : "Pending,"),
          obj->pipeline_stats.bo ? "yes" : "no");
      break;
   }
}

void
brw_dump_perf_queries(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;
   DBG("Queries: (Open queries = %d, OA users = %d)\n",
       brw->perfquery.n_active_oa_queries, brw->perfquery.n_oa_users);
   _mesa_HashWalk(ctx->PerfQuery.Objects, dump_perf_query_callback, brw);
}

/******************************************************************************/

static void
brw_get_perf_query_info(struct gl_context *ctx,
                        int query_index,
                        const char **name,
                        GLuint *data_size,
                        GLuint *n_counters,
                        GLuint *n_active)
{
   struct brw_context *brw = brw_context(ctx);
   const struct brw_perf_query *query = &brw->perfquery.queries[query_index];

   *name = query->name;
   *data_size = query->data_size;
   *n_counters = query->n_counters;

   switch(query->kind) {
   case OA_COUNTERS:
      *n_active = brw->perfquery.n_active_oa_queries;
      break;

   case PIPELINE_STATS:
      *n_active = brw->perfquery.n_active_pipeline_stats_queries;
      break;
   }
}

static void
brw_get_perf_counter_info(struct gl_context *ctx,
                          int query_index,
                          int counter_index,
                          const char **name,
                          const char **desc,
                          GLuint *offset,
                          GLuint *data_size,
                          GLuint *type_enum,
                          GLuint *data_type_enum,
                          GLuint64 *raw_max)
{
   struct brw_context *brw = brw_context(ctx);
   const struct brw_perf_query *query = &brw->perfquery.queries[query_index];
   const struct brw_perf_query_counter *counter =
      &query->counters[counter_index];

   *name = counter->name;
   *desc = counter->desc;
   *offset = counter->offset;
   *data_size = counter->size;
   *type_enum = counter->type;
   *data_type_enum = counter->data_type;
   *raw_max = counter->raw_max;
}

/**
 * Take a snapshot of any queryed pipeline statistics counters.
 */
static void
snapshot_statistics_registers(struct brw_context *brw,
                              struct brw_perf_query_object *obj,
                              uint32_t offset_in_bytes)
{
   const int offset = offset_in_bytes / sizeof(uint64_t);
   const struct brw_perf_query *query = obj->query;
   const int n_counters = query->n_counters;

   intel_batchbuffer_emit_mi_flush(brw);

   for (int i = 0; i < n_counters; i++) {
      const struct brw_perf_query_counter *counter = &query->counters[i];

      assert(counter->data_type == GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL);

      brw_store_register_mem64(brw, obj->pipeline_stats.bo,
                               counter->pipeline_stat_reg,
                               offset + i);
   }
}

/**
 * Gather results from pipeline_stats_bo, storing the final values.
 *
 * This allows us to free pipeline_stats_bo (which is 4K) in favor of a much
 * smaller array of final results.
 */
static void
gather_statistics_results(struct brw_context *brw,
                          struct brw_perf_query_object *obj)
{
   const int n_counters = obj->query->n_counters;

   obj->pipeline_stats.results = calloc(n_counters, sizeof(uint64_t));
   if (obj->pipeline_stats.results == NULL) {
      _mesa_error_no_memory(__func__);
      return;
   }

   drm_intel_bo_map(obj->pipeline_stats.bo, false);
   uint64_t *start = obj->pipeline_stats.bo->virtual;
   uint64_t *end = start + (SECOND_SNAPSHOT_OFFSET_IN_BYTES / sizeof(uint64_t));

   for (int i = 0; i < n_counters; i++)
      obj->pipeline_stats.results[i] = end[i] - start[i];

   drm_intel_bo_unmap(obj->pipeline_stats.bo);
   drm_intel_bo_unreference(obj->pipeline_stats.bo);
   obj->pipeline_stats.bo = NULL;
}

/******************************************************************************/

/**
 * Emit an MI_REPORT_PERF_COUNT command packet.
 *
 * This writes the current OA counter values to buffer.
 */
static void
emit_mi_report_perf_count(struct brw_context *brw,
                          drm_intel_bo *bo,
                          uint32_t offset_in_bytes,
                          uint32_t report_id)
{
   assert(offset_in_bytes % 64 == 0);

   /* Reports apparently don't always get written unless we flush first. */
   intel_batchbuffer_emit_mi_flush(brw);

   if (brw->gen == 7) {
      BEGIN_BATCH(3);
      OUT_BATCH(GEN6_MI_REPORT_PERF_COUNT);
      OUT_RELOC(bo, I915_GEM_DOMAIN_INSTRUCTION, I915_GEM_DOMAIN_INSTRUCTION,
                offset_in_bytes);
      OUT_BATCH(report_id);
      ADVANCE_BATCH();
   } else
      unreachable("Unsupported generation for OA performance counters.");

   /* Reports apparently don't always get written unless we flush after. */
   intel_batchbuffer_emit_mi_flush(brw);
}

static unsigned int
read_perf_head(struct perf_event_mmap_page *mmap_page)
{
   unsigned int head = (*(volatile uint64_t *)&mmap_page->data_head);
   rmb();

   return head;
}

static void
write_perf_tail(struct perf_event_mmap_page *mmap_page,
                unsigned int tail)
{
   /* Make sure we've finished reading all the sample data we
    * we're consuming before updating the tail... */
   mb();
   mmap_page->data_tail = tail;
}

/* Update the real perf tail pointer according to the query tail that
 * is currently furthest behind...
 */
static void
update_perf_tail(struct brw_context *brw)
{
   unsigned int size = brw->perfquery.perf_oa_buffer_size;
   unsigned int head = read_perf_head(brw->perfquery.perf_oa_mmap_page);
   int straggler_taken = -1;
   unsigned int straggler_tail;

   for (int i = 0; i < brw->perfquery.unresolved_elements; i++) {
      struct brw_perf_query_object *obj = brw->perfquery.unresolved[i];
      int taken;

      if (!obj->oa.bo)
         continue;

      taken = TAKEN(head, obj->oa.perf_tail, size);

      if (taken > straggler_taken) {
         straggler_taken = taken;
         straggler_tail = obj->oa.perf_tail;
      }
   }

   if (straggler_taken >= 0)
      write_perf_tail(brw->perfquery.perf_oa_mmap_page, straggler_tail);
}

/**
 * Add a query to the global list of "unresolved queries."
 *
 * Queries are "unresolved" until all the counter snapshots have been
 * accumulated via accumulate_oa_snapshots() after the end MI_REPORT_PERF_COUNT
 * has landed in query->oa.bo.
 */
static void
add_to_unresolved_query_list(struct brw_context *brw,
                             struct brw_perf_query_object *obj)
{
   if (brw->perfquery.unresolved_elements >=
       brw->perfquery.unresolved_array_size) {
      brw->perfquery.unresolved_array_size *= 1.5;
      brw->perfquery.unresolved = reralloc(brw, brw->perfquery.unresolved,
                                           struct brw_perf_query_object *,
                                           brw->perfquery.unresolved_array_size);
   }

   brw->perfquery.unresolved[brw->perfquery.unresolved_elements++] = obj;

   if (obj->oa.bo)
      update_perf_tail(brw);
}

/**
 * Remove a query from the global list of "unresolved queries." once
 * the end MI_RPC OA counter snapshot has been accumulated, or when
 * discarding unwanted query results.
 */
static void
drop_from_unresolved_query_list(struct brw_context *brw,
                                struct brw_perf_query_object *obj)
{
   for (int i = 0; i < brw->perfquery.unresolved_elements; i++) {
      if (brw->perfquery.unresolved[i] == obj) {
         int last_elt = --brw->perfquery.unresolved_elements;

         if (i == last_elt)
            brw->perfquery.unresolved[i] = NULL;
         else
            brw->perfquery.unresolved[i] = brw->perfquery.unresolved[last_elt];

         break;
      }
   }

   if (obj->oa.bo)
      update_perf_tail(brw);
}

static int
get_eu_count(uint32_t devid)
{
   const struct brw_device_info *info = brw_get_device_info(devid);

   assert(info && info->is_haswell);

   if (info->gt == 1)
      return 10;
   else if (info->gt == 2)
      return 20;
   else if (info->gt == 3)
      return 40;

   unreachable("Unexpected Haswell GT number");
}

static uint64_t
read_report_timestamp(struct brw_context *brw, uint32_t *report)
{
   return brw->perfquery.read_oa_report_timestamp(report);
}

/**
 * Given pointers to starting and ending OA snapshots, add the deltas for each
 * counter to the results.
 */
static void
add_deltas(struct brw_context *brw,
           struct brw_perf_query_object *obj,
           uint32_t *start, uint32_t *end)
{
   const struct brw_perf_query *query = obj->query;

#if 0
   fprintf(stderr, "Accumulating delta:\n");
   fprintf(stderr, "> Start timestamp = %" PRIu64 "\n", read_report_timestamp(brw, start));
   fprintf(stderr, "> End timestamp = %" PRIu64 "\n", read_report_timestamp(brw, end));
#endif

   for (int i = 0; i < query->n_oa_counters; i++) {
      struct brw_oa_counter *oa_counter = &query->oa_counters[i];
      //uint64_t pre_accumulate;

      if (!oa_counter->accumulate)
         continue;

      //pre_accumulate = query->oa.accumulator[counter->id];
      oa_counter->accumulate(oa_counter,
                             start, end,
                             obj->oa.accumulator);
#if 0
      fprintf(stderr, "> Updated %s from %" PRIu64 " to %" PRIu64 "\n",
              counter->name, pre_accumulate,
              query->oa.accumulator[counter->id]);
#endif
   }
}

/* Handle restarting ioctl if interupted... */
static int
perf_ioctl(int fd, unsigned long request, void *arg)
{
   int ret;

   do {
      ret = ioctl(fd, request, arg);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
   return ret;
}

static bool
inc_n_oa_users(struct brw_context *brw)
{
   if (brw->perfquery.n_oa_users == 0 &&
       perf_ioctl(brw->perfquery.perf_oa_event_fd,
                  PERF_EVENT_IOC_ENABLE, 0) < 0)
   {
      return false;
   }
   ++brw->perfquery.n_oa_users;

   return true;
}

static void
dec_n_oa_users(struct brw_context *brw)
{
   /* Disabling the i915_oa event will effectively disable the OA
    * counters.  Note it's important to be sure there are no outstanding
    * MI_RPC commands at this point since they could stall the CS
    * indefinitely once OACONTROL is disabled.
    */
   --brw->perfquery.n_oa_users;
   if (brw->perfquery.n_oa_users == 0 &&
       perf_ioctl(brw->perfquery.perf_oa_event_fd,
                  PERF_EVENT_IOC_DISABLE, 0) < 0)
   {
      DBG("WARNING: Error disabling i915_oa perf event: %m\n");
   }
}

/**
 * Accumulate OA counter results from a series of snapshots.
 *
 * N.B. We write snapshots for the beginning and end of a query into
 * query->oa.bo as well as collect periodic snapshots from the Linux
 * perf interface.
 *
 * These periodic snapshots help to ensure we handle counter overflow
 * correctly by being frequent enough to ensure we don't miss multiple
 * wrap overflows of a counter between snapshots.
 */
static void
accumulate_oa_snapshots(struct brw_context *brw,
                        struct brw_perf_query_object *obj)
{
   struct gl_perf_query_object *o = &obj->base;
   uint32_t *query_buffer;
   uint8_t *data = brw->perfquery.perf_oa_mmap_base + brw->perfquery.page_size;
   const unsigned int size = brw->perfquery.perf_oa_buffer_size;
   const uint64_t mask = size - 1;
   uint64_t head;
   uint64_t tail;
   uint32_t *start;
   uint64_t start_timestamp;
   uint32_t *last;
   uint32_t *end;
   uint64_t end_timestamp;
   uint8_t scratch[MAX_OA_PERF_SAMPLE_SIZE];

   assert(o->Ready);

   if (perf_ioctl(brw->perfquery.perf_oa_event_fd,
                  PERF_EVENT_IOC_FLUSH, 0) < 0)
      DBG("Failed to flush outstanding perf events: %m\n");

   drm_intel_bo_map(obj->oa.bo, false);
   query_buffer = obj->oa.bo->virtual;

   start = last = query_buffer;
   end = query_buffer + (SECOND_SNAPSHOT_OFFSET_IN_BYTES / sizeof(uint32_t));

#warning "TODO: find a way to report OA errors from the kernel"
   /* XXX: Is there anything we can do to handle this gracefully/
    * report the error to the application? */
   if (start[0] != obj->oa.current_report_id)
      DBG("Spurious start report id=%"PRIu32"\n", start[0]);
   if (end[0] != (obj->oa.current_report_id + 1))
      DBG("Spurious end report id=%"PRIu32"\n", start[0]);

   start_timestamp = read_report_timestamp(brw, start);
   end_timestamp = read_report_timestamp(brw, end);

   head = read_perf_head(brw->perfquery.perf_oa_mmap_page);
   tail = obj->oa.perf_tail;

   //fprintf(stderr, "Handle event mask = 0x%" PRIx64
   //        " head=%" PRIu64 " tail=%" PRIu64 "\n", mask, head, tail);

   while (TAKEN(head, tail, size)) {
      const struct perf_event_header *header =
         (const struct perf_event_header *)(data + (tail & mask));

      if (header->size == 0) {
         DBG("Spurious header size == 0\n");
         /* XXX: How should we handle this instead of exiting() */
#warning "FIXME: avoid exit(1) in error condition"
         exit(1);
      }

      if (header->size > (head - tail)) {
         DBG("Spurious header size would overshoot head\n");
         /* XXX: How should we handle this instead of exiting() */
         exit(1);
      }

      //fprintf(stderr, "header = %p tail=%" PRIu64 " size=%d\n",
      //        header, tail, header->size);

      if ((const uint8_t *)header + header->size > data + size) {
         int before;

         if (header->size > MAX_OA_PERF_SAMPLE_SIZE) {
            DBG("Skipping spurious sample larger than expected\n");
            tail += header->size;
            continue;
         }

         before = data + size - (const uint8_t *)header;

         memcpy(scratch, header, before);
         memcpy(scratch + before, data, header->size - before);

         header = (struct perf_event_header *)scratch;
         //fprintf(stderr, "DEBUG: split\n");
         //exit(1);
      }

      switch (header->type) {
         case PERF_RECORD_LOST: {
            struct {
               struct perf_event_header header;
               uint64_t id;
               uint64_t n_lost;
            } *lost = (void *)header;
            DBG("i915_oa: Lost %" PRIu64 " events\n", lost->n_lost);
            break;
         }

         case PERF_RECORD_THROTTLE:
            DBG("i915_oa: Sampling has been throttled\n");
            break;

         case PERF_RECORD_UNTHROTTLE:
            DBG("i915_oa: Sampling has been unthrottled\n");
            break;

         case PERF_RECORD_SAMPLE: {
            struct oa_perf_sample *perf_sample = (struct oa_perf_sample *)header;
            uint32_t *report = (uint32_t *)perf_sample->raw_data;
            uint64_t timestamp = read_report_timestamp(brw, report);

            if (timestamp >= end_timestamp)
               goto end;

            if (timestamp > start_timestamp) {
               add_deltas(brw, obj, last, report);
               last = report;
            }

            break;
         }

         default:
            DBG("i915_oa: Spurious header type = %d\n", header->type);
      }

      //fprintf(stderr, "Tail += %d\n", header->size);

      tail += header->size;
   }

end:

   add_deltas(brw, obj, last, end);

   DBG("Marking %d resolved - results gathered\n", o->Id);

   drm_intel_bo_unmap(obj->oa.bo);
   obj->oa.results_accumulated = true;
   drop_from_unresolved_query_list(brw, obj);
   dec_n_oa_users(brw);
}

/******************************************************************************/

static uint64_t
read_file_uint64 (const char *file)
{
   char buf[32];
   int fd, n;

   fd = open(file, 0);
   if (fd < 0)
      return 0;
   n = read(fd, buf, sizeof (buf) - 1);
   close(fd);
   if (n < 0)
      return 0;

   buf[n] = '\0';
   return strtoull(buf, 0, 0);
}

static uint64_t
lookup_i915_oa_id (void)
{
   return read_file_uint64("/sys/bus/event_source/devices/i915_oa/type");
}

static long
perf_event_open (struct perf_event_attr *hw_event,
                 pid_t pid,
                 int cpu,
                 int group_fd,
                 unsigned long flags)
{
   return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static bool
open_i915_oa_event(struct brw_context *brw,
                   int counter_profile_id,
                   uint64_t report_format,
                   int period_exponent,
                   int drm_fd,
                   uint64_t ctx_id)
{
   struct perf_event_attr attr;
   int event_fd;
   void *mmap_base;

   memset(&attr, 0, sizeof (struct perf_event_attr));
   attr.size = sizeof (struct perf_event_attr);
   attr.type = lookup_i915_oa_id();

   attr.config |= (uint64_t)counter_profile_id << I915_PERF_OA_PROFILE_SHIFT;
   attr.config |= report_format;
   attr.config |= (uint64_t)period_exponent << I915_PERF_OA_TIMER_EXPONENT_SHIFT;

   attr.config |= I915_PERF_OA_SINGLE_CONTEXT_ENABLE;
   attr.config |= ctx_id & I915_PERF_OA_CTX_ID_MASK;
   attr.config1 = drm_fd;

   attr.sample_type = PERF_SAMPLE_RAW;
   attr.disabled = 1;
   attr.sample_period = 1;

   event_fd = perf_event_open(&attr,
                              -1, /* pid */
                              0, /* cpu */
                              -1, /* group fd */
                              PERF_FLAG_FD_CLOEXEC); /* flags */
   if (event_fd == -1) {
      DBG("Error opening i915_oa perf event: %m\n");
      return false;
   }

   /* NB: A read-write mapping ensures the kernel will stop writing data when
    * the buffer is full, and will report samples as lost. */
   mmap_base = mmap(NULL,
                    brw->perfquery.perf_oa_buffer_size + brw->perfquery.page_size,
                    PROT_READ | PROT_WRITE, MAP_SHARED, event_fd, 0);
   if (mmap_base == MAP_FAILED) {
      DBG("Error mapping circular buffer, %m\n");
      close (event_fd);
      return false;
   }

   brw->perfquery.perf_oa_event_fd = event_fd;
   brw->perfquery.perf_oa_mmap_base = mmap_base;
   brw->perfquery.perf_oa_mmap_page = mmap_base;

   brw->perfquery.perf_profile_id = counter_profile_id;
   brw->perfquery.perf_oa_format_id = report_format;

   return true;
}

/**
 * Driver hook for glBeginPerfQueryINTEL().
 */
static GLboolean
brw_begin_perf_query(struct gl_context *ctx,
                     struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);
   const struct brw_perf_query *query = obj->query;

   assert(!o->Active);
   assert(!o->Used || o->Ready); /* no in-flight query to worry about */

   DBG("Begin(%d)\n", o->Id);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      /* If the OA counters aren't already on, enable them. */
      if (brw->perfquery.perf_oa_event_fd == -1) {
         __DRIscreen *screen = brw->intelScreen->driScrnPriv;
         uint64_t ctx_id = drm_intel_gem_context_get_context_id(brw->hw_ctx);
         int period_exponent;

         /* The timestamp for HSW+ increments every 80ns
          *
          * The period_exponent gives a sampling period as follows:
          *   sample_period = 80ns * 2^(period_exponent + 1)
          *
          * The overflow period for Haswell can be calculated as:
          *
          * 2^32 / (n_eus * max_gen_freq * 2)
          * (E.g. 40 EUs @ 1GHz = ~53ms)
          *
          * We currently sample every 42 milliseconds...
          */
         period_exponent = 18;

         if (!open_i915_oa_event(brw,
                                 query->perf_profile_id,
                                 query->perf_oa_format_id,
                                 period_exponent,
                                 screen->fd, /* drm fd */
                                 ctx_id))
            return GL_FALSE;
      } else {
         /* Opening an i915_oa event fd implies exclusive access to
          * the OA unit which will generate counter reports for a
          * specific counter set/profile with a specific layout/format
          * so we can't begin any OA based queries that require a
          * different profile or format unless we get an opportunity
          * to close the event fd and open a new one...
          */
         if (brw->perfquery.perf_profile_id != query->perf_profile_id ||
             brw->perfquery.perf_oa_format_id != query->perf_oa_format_id)
         {
            return false;
         }
      }

      if (!inc_n_oa_users(brw)) {
         DBG("WARNING: Error enabling i915_oa perf event: %m\n");
         return GL_FALSE;
      }

      if (obj->oa.bo) {
         drm_intel_bo_unreference(obj->oa.bo);
         obj->oa.bo = NULL;
      }

      obj->oa.bo =
         drm_intel_bo_alloc(brw->bufmgr, "perf. query OA bo", 4096, 64);
#ifdef DEBUG
      /* Pre-filling the BO helps debug whether writes landed. */
      drm_intel_bo_map(obj->oa.bo, true);
      memset((char *) obj->oa.bo->virtual, 0x80, 4096);
      drm_intel_bo_unmap(obj->oa.bo);
#endif

      obj->oa.current_report_id = brw->perfquery.next_query_start_report_id;
      brw->perfquery.next_query_start_report_id += 2;

      /* Take a starting OA counter snapshot. */
      emit_mi_report_perf_count(brw, obj->oa.bo, 0,
                                obj->oa.current_report_id);
      ++brw->perfquery.n_active_oa_queries;

      /* Each unresolved query maintains a separate tail pointer into the
       * circular perf sample buffer. The real tail pointer in
       * perfquery.perf_oa_mmap_page.data_tail will correspond to the query
       * tail that is furthest behind.
       */
      obj->oa.perf_tail = read_perf_head(brw->perfquery.perf_oa_mmap_page);

      memset(obj->oa.accumulator, 0, sizeof(obj->oa.accumulator));
      obj->oa.results_accumulated = false;

      add_to_unresolved_query_list(brw, obj);
      break;

   case PIPELINE_STATS:
      if (obj->pipeline_stats.bo) {
         drm_intel_bo_unreference(obj->pipeline_stats.bo);
         obj->pipeline_stats.bo = NULL;
      }

      obj->pipeline_stats.bo =
         drm_intel_bo_alloc(brw->bufmgr, "perf. query stats bo", 4096, 64);

      /* Take starting snapshots. */
      snapshot_statistics_registers(brw, obj, 0);

      free(obj->pipeline_stats.results);
      obj->pipeline_stats.results = NULL;

      ++brw->perfquery.n_active_pipeline_stats_queries;
      break;
   }

   return true;
}

/**
 * Driver hook for glEndPerfQueryINTEL().
 */
static void
brw_end_perf_query(struct gl_context *ctx,
                     struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   DBG("End(%d)\n", o->Id);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      /* Take an ending OA counter snapshot. */
      emit_mi_report_perf_count(brw, obj->oa.bo,
                                SECOND_SNAPSHOT_OFFSET_IN_BYTES,
                                obj->oa.current_report_id + 1);
      --brw->perfquery.n_active_oa_queries;

      /* NB: even though the query has now ended, it can't be resolved
       * until the end MI_REPORT_PERF_COUNT snapshot has been written
       * to query->oa.bo */
      break;

   case PIPELINE_STATS:
      /* Take ending snapshots. */
      snapshot_statistics_registers(brw, obj,
                                    SECOND_SNAPSHOT_OFFSET_IN_BYTES);
      --brw->perfquery.n_active_pipeline_stats_queries;
      break;
   }
}

static void
brw_wait_perf_query(struct gl_context *ctx, struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);
   drm_intel_bo *bo = NULL;

   assert(!o->Ready);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      bo = obj->oa.bo;
      break;

   case PIPELINE_STATS:
      bo = obj->pipeline_stats.bo;
      break;
   }

   if (bo == NULL)
      return;

   /* If the current batch references our results bo then we need to
    * flush first... */
   if (drm_intel_bo_references(brw->batch.bo, bo))
      intel_batchbuffer_flush(brw);

   if (unlikely(brw->perf_debug)) {
      if (drm_intel_bo_busy(bo))
         perf_debug("Stalling GPU waiting for a performance query object.\n");
   }

   drm_intel_bo_wait_rendering(bo);
}

/**
 * Is a performance query result available?
 */
static GLboolean
brw_is_perf_query_ready(struct gl_context *ctx,
                        struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   if (o->Ready)
      return true;

   switch(obj->query->kind) {
   case OA_COUNTERS:
      return (obj->oa.results_accumulated ||
              (obj->oa.bo &&
               !drm_intel_bo_references(brw->batch.bo, obj->oa.bo) &&
               !drm_intel_bo_busy(obj->oa.bo)));

   case PIPELINE_STATS:
      return (obj->pipeline_stats.bo &&
              !drm_intel_bo_references(brw->batch.bo, obj->pipeline_stats.bo) &&
              !drm_intel_bo_busy(obj->pipeline_stats.bo));
   }

   unreachable("missing ready check for unknown query kind");
   return false;
}

static int
get_oa_counter_data(struct brw_context *brw,
                    struct brw_perf_query_object *obj,
                    size_t data_size,
                    uint8_t *data)
{
   const struct brw_perf_query *query = obj->query;
   int n_counters = query->n_counters;
   int written = 0;

   if (!obj->oa.results_accumulated) {
      accumulate_oa_snapshots(brw, obj);
      assert(obj->oa.results_accumulated);
   }

   for (int i = 0; i < n_counters; i++) {
      const struct brw_perf_query_counter *counter = &query->counters[i];

      if (counter->size) {
         counter->oa_counter->read(counter->oa_counter, obj->oa.accumulator,
                                   data + counter->offset);
         written = counter->offset + counter->size;
      }
   }

   return written;
}

static int
get_pipeline_stats_data(struct brw_context *brw,
                        struct brw_perf_query_object *obj,
                        size_t data_size,
                        uint8_t *data)

{
   int n_counters = obj->query->n_counters;
   uint8_t *p = data;

   if (!obj->pipeline_stats.results) {
      gather_statistics_results(brw, obj);

      /* Check if we did really get the results */
      if (!obj->pipeline_stats.results)
         return 0;
   }

   for (int i = 0; i < n_counters; i++) {
      *((uint64_t *)p) = obj->pipeline_stats.results[i];
      p += 8;
   }

   return p - data;
}

/**
 * Get the performance query result.
 */
static void
brw_get_perf_query_data(struct gl_context *ctx,
                        struct gl_perf_query_object *o,
                        GLsizei data_size,
                        GLuint *data,
                        GLuint *bytes_written)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);
   int written = 0;

   assert(brw_is_perf_query_ready(ctx, o));

   DBG("GetData(%d)\n", o->Id);
   brw_dump_perf_queries(brw);

   /* This hook should only be called when results are available. */
   assert(o->Ready);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      written = get_oa_counter_data(brw, obj, data_size, (uint8_t *)data);
      break;

   case PIPELINE_STATS:
      written = get_pipeline_stats_data(brw, obj, data_size, (uint8_t *)data);
      break;
   }

   if (bytes_written)
      *bytes_written = written;
}

/**
 * Create a new performance query object.
 */
static struct gl_perf_query_object *
brw_create_perf_query(struct gl_context *ctx, int query_index)
{
   struct brw_context *brw = brw_context(ctx);
   const struct brw_perf_query *query = &brw->perfquery.queries[query_index];
   struct brw_perf_query_object *obj =
      calloc(1, sizeof(struct brw_perf_query_object));

   if (!obj)
      return NULL;

   obj->query = query;

   brw->perfquery.n_query_instances++;

   return &obj->base;
}

static void
close_perf(struct brw_context *brw)
{
   if (brw->perfquery.perf_oa_event_fd != -1) {
      if (brw->perfquery.perf_oa_mmap_base) {
         size_t mapping_len =
            brw->perfquery.perf_oa_buffer_size + brw->perfquery.page_size;

         munmap(brw->perfquery.perf_oa_mmap_base, mapping_len);
         brw->perfquery.perf_oa_mmap_base = NULL;
      }

      close(brw->perfquery.perf_oa_event_fd);
      brw->perfquery.perf_oa_event_fd = -1;
   }
}

/**
 * Delete a performance query object.
 */
static void
brw_delete_perf_query(struct gl_context *ctx,
                      struct gl_perf_query_object *o)
{
   struct brw_context *brw = brw_context(ctx);
   struct brw_perf_query_object *obj = brw_perf_query(o);

   assert(!o->Active);
   assert(!o->Used || o->Ready); /* no in-flight query to worry about */

   DBG("Delete(%d)\n", o->Id);

   switch(obj->query->kind) {
   case OA_COUNTERS:
      if (obj->oa.bo) {
         if (!obj->oa.results_accumulated) {
            drop_from_unresolved_query_list(brw, obj);
            dec_n_oa_users(brw);
         }

         drm_intel_bo_unreference(obj->oa.bo);
         obj->oa.bo = NULL;
      }

      obj->oa.results_accumulated = false;
      break;

   case PIPELINE_STATS:
      if (obj->pipeline_stats.bo) {
         drm_intel_bo_unreference(obj->pipeline_stats.bo);
         obj->pipeline_stats.bo = NULL;
      }

      free(obj->pipeline_stats.results);
      obj->pipeline_stats.results = NULL;
      break;
   }

   free(obj);

   if (--brw->perfquery.n_query_instances == 0)
      close_perf(brw);
}

/******************************************************************************/

/* Type safe wrappers for reading OA counter values */

static uint64_t
read_uint64_oa_counter(struct brw_oa_counter *counter, uint64_t *accumulated)
{
   uint64_t value;

   assert(counter->data_type == GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL);

   counter->read(counter, accumulated, &value);

   return value;
}

static float
read_float_oa_counter(struct brw_oa_counter *counter, uint64_t *accumulated)
{
   float value;

   assert(counter->data_type == GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL);

   counter->read(counter, accumulated, &value);

   return value;
}

/******************************************************************************/

/*
 * OA counter normalisation support...
 */

static void
read_accumulated_oa_counter_cb(struct brw_oa_counter *counter,
                               uint64_t *accumulator,
                               void *value)
{
   *((uint64_t *)value) = accumulator[counter->accumulator_index];
}

static void
accumulate_uint32_cb(struct brw_oa_counter *counter,
                     uint32_t *report0,
                     uint32_t *report1,
                     uint64_t *accumulator)
{
   accumulator[counter->accumulator_index] +=
      (uint32_t)(report1[counter->report_offset] -
                 report0[counter->report_offset]);
}

#if 0
/* XXX: we should factor this out for now, but notably BDW has 40bit counters... */
static void
accumulate_uint40_cb(struct brw_oa_counter *counter,
                     uint32_t *report0,
                     uint32_t *report1,
                     uint64_t *accumulator)
{
   uint32_t value0 = report0[counter->report_offset];
   uint32_t value1 = report1[counter->report_offset];
   uint64_t delta;

   if (value0 > value1)
      delta = (1ULL << 40) + value1 - value0;
   else
      delta = value1 - value0;

   accumulator[counter->accumulator_index] += delta;
}
#endif

static struct brw_oa_counter *
add_raw_oa_counter(struct brw_query_builder *builder, int report_offset)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->report_offset = report_offset;
   counter->accumulator_index = builder->next_accumulator_index++;
   counter->accumulate = accumulate_uint32_cb;
   counter->read = read_accumulated_oa_counter_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static uint64_t
hsw_read_report_timestamp(uint32_t *report)
{
   /* The least significant timestamp bit represents 80ns on Haswell */
   return ((uint64_t)report[1]) * 80;
}

static void
accumulate_hsw_elapsed_cb(struct brw_oa_counter *counter,
                          uint32_t *report0,
                          uint32_t *report1,
                          uint64_t *accumulator)
{
   uint64_t timestamp0 = hsw_read_report_timestamp(report0);
   uint64_t timestamp1 = hsw_read_report_timestamp(report1);

   accumulator[counter->accumulator_index] += (timestamp1 - timestamp0);
}

static struct brw_oa_counter *
add_hsw_elapsed_oa_counter(struct brw_query_builder *builder)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->accumulator_index = builder->next_accumulator_index++;
   counter->accumulate = accumulate_hsw_elapsed_cb;
   counter->read = read_accumulated_oa_counter_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_frequency_cb(struct brw_oa_counter *counter,
                  uint64_t *accumulated,
                  void *value) /* uint64 */
{
   uint64_t clk_delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t time_delta = read_uint64_oa_counter(counter->reference1, accumulated);
   uint64_t *ret = value;

   if (!clk_delta) {
      *ret = 0;
      return;
   }

   *ret = (clk_delta * 1000) / time_delta;
}

static struct brw_oa_counter *
add_avg_frequency_oa_counter(struct brw_query_builder *builder,
                             struct brw_oa_counter *timestamp)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   assert(timestamp->data_type == GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL);

   counter->reference0 = builder->gpu_core_clock;
   counter->reference1 = timestamp;
   counter->read = read_frequency_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_oa_counter_normalized_by_gpu_duration_cb(struct brw_oa_counter *counter,
                                              uint64_t *accumulated,
                                              void *value) /* float */
{
   uint64_t delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t clk_delta = read_uint64_oa_counter(counter->reference1, accumulated);
   float *ret = value;

   if (!clk_delta) {
      *ret = 0;
      return;
   }

   *ret = ((double)delta * 100.0) / (double)clk_delta;
}

static struct brw_oa_counter *
add_oa_counter_normalised_by_gpu_duration(struct brw_query_builder *builder,
                                          struct brw_oa_counter *raw)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = raw;
   counter->reference1 = builder->gpu_core_clock;
   counter->read = read_oa_counter_normalized_by_gpu_duration_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL;

   return counter;
}

static void
read_hsw_samplers_busy_duration_cb(struct brw_oa_counter *counter,
                                   uint64_t *accumulated,
                                   void *value) /* float */
{
   uint64_t sampler0_busy = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t sampler1_busy = read_uint64_oa_counter(counter->reference1, accumulated);
   uint64_t clk_delta = read_uint64_oa_counter(counter->reference2, accumulated);
   float *ret = value;

   if (!clk_delta) {
      *ret = 0;
      return;
   }

   *ret = ((double)(sampler0_busy + sampler1_busy) * 100.0) / ((double)clk_delta * 2.0);
}

static struct brw_oa_counter *
add_hsw_samplers_busy_duration_oa_counter(struct brw_query_builder *builder,
                                          struct brw_oa_counter *sampler0_busy_raw,
                                          struct brw_oa_counter *sampler1_busy_raw)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = sampler0_busy_raw;
   counter->reference1 = sampler1_busy_raw;
   counter->reference2 = builder->gpu_core_clock;
   counter->read = read_hsw_samplers_busy_duration_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL;

   return counter;
}

static void
read_hsw_slice_extrapolated_cb(struct brw_oa_counter *counter,
                               uint64_t *accumulated,
                               void *value) /* float */
{
   uint64_t counter0 = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t counter1 = read_uint64_oa_counter(counter->reference1, accumulated);
   int eu_count = counter->config;
   uint64_t *ret = value;

   *ret = (counter0 + counter1) * eu_count;
}

static struct brw_oa_counter *
add_hsw_slice_extrapolated_oa_counter(struct brw_query_builder *builder,
                                      struct brw_oa_counter *counter0,
                                      struct brw_oa_counter *counter1)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = counter0;
   counter->reference1 = counter1;
   counter->config = builder->brw->perfquery.eu_count;
   counter->read = read_hsw_slice_extrapolated_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_oa_counter_normalized_by_eu_duration_cb(struct brw_oa_counter *counter,
                                             uint64_t *accumulated,
                                             void *value) /* float */
{
   uint64_t delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t clk_delta = read_uint64_oa_counter(counter->reference1, accumulated);
   float *ret = value;

   if (!clk_delta) {
      *ret = 0;
      return;
   }

   delta /= counter->config; /* EU count */

   *ret = (double)delta * 100.0 / (double)clk_delta;
}

static struct brw_oa_counter *
add_oa_counter_normalised_by_eu_duration(struct brw_query_builder *builder,
                                         struct brw_oa_counter *raw)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = raw;
   counter->reference1 = builder->gpu_core_clock;
   counter->config = builder->brw->perfquery.eu_count;
   counter->read = read_oa_counter_normalized_by_eu_duration_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL;

   return counter;
}

static void
read_av_thread_cycles_counter_cb(struct brw_oa_counter *counter,
                                 uint64_t *accumulated,
                                 void *value) /* uint64 */
{
   uint64_t delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t spawned = read_uint64_oa_counter(counter->reference1, accumulated);
   uint64_t *ret = value;

   if (!spawned) {
      *ret = 0;
      return;
   }

   *ret = delta / spawned;
}

static struct brw_oa_counter *
add_average_thread_cycles_oa_counter(struct brw_query_builder *builder,
                                     struct brw_oa_counter *raw,
                                     struct brw_oa_counter *denominator)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = raw;
   counter->reference1 = denominator;
   counter->read = read_av_thread_cycles_counter_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_scaled_uint64_counter_cb(struct brw_oa_counter *counter,
                              uint64_t *accumulated,
                              void *value) /* uint64 */
{
   uint64_t delta = read_uint64_oa_counter(counter->reference0, accumulated);
   uint64_t scale = counter->config;
   uint64_t *ret = value;

   *ret = delta * scale;
}

static struct brw_oa_counter *
add_scaled_uint64_oa_counter(struct brw_query_builder *builder,
                             struct brw_oa_counter *input,
                             int scale)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = input;
   counter->config = scale;
   counter->read = read_scaled_uint64_counter_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;

   return counter;
}

static void
read_max_of_float_counters_cb(struct brw_oa_counter *counter,
                              uint64_t *accumulated,
                              void *value) /* float */
{
   float counter0 = read_float_oa_counter(counter->reference0, accumulated);
   float counter1 = read_float_oa_counter(counter->reference1, accumulated);
   float *ret = value;

   *ret = counter0 >= counter1 ? counter0 : counter1;
}


static struct brw_oa_counter *
add_max_of_float_oa_counters(struct brw_query_builder *builder,
                             struct brw_oa_counter *counter0,
                             struct brw_oa_counter *counter1)
{
   struct brw_oa_counter *counter =
      &builder->query->oa_counters[builder->query->n_oa_counters++];

   counter->reference0 = counter0;
   counter->reference1 = counter1;
   counter->read = read_max_of_float_counters_cb;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL;

   return counter;
}

static void
report_uint64_oa_counter_as_raw_uint64(struct brw_query_builder *builder,
                                       const char *name,
                                       const char *desc,
                                       struct brw_oa_counter *oa_counter)
{
   struct brw_perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = GL_PERFQUERY_COUNTER_RAW_INTEL;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;
   counter->raw_max = 0; /* undefined range */
   counter->offset = pot_align(builder->offset, 8);
   counter->size = sizeof(uint64_t);

   builder->offset = counter->offset + counter->size;
}

static void
report_uint64_oa_counter_as_uint64_event(struct brw_query_builder *builder,
                                         const char *name,
                                         const char *desc,
                                         struct brw_oa_counter *oa_counter)
{
   struct brw_perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = GL_PERFQUERY_COUNTER_EVENT_INTEL;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;
   counter->offset = pot_align(builder->offset, 8);
   counter->size = sizeof(uint64_t);

   builder->offset = counter->offset + counter->size;
}

static void
report_float_oa_counter_as_percentage_duration(struct brw_query_builder *builder,
                                               const char *name,
                                               const char *desc,
                                               struct brw_oa_counter *oa_counter)
{
   struct brw_perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = GL_PERFQUERY_COUNTER_DURATION_RAW_INTEL;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_FLOAT_INTEL;
   counter->raw_max = 100;
   counter->offset = pot_align(builder->offset, 4);
   counter->size = sizeof(float);

   builder->offset = counter->offset + counter->size;
}

static void
report_uint64_oa_counter_as_throughput(struct brw_query_builder *builder,
                                       const char *name,
                                       const char *desc,
                                       struct brw_oa_counter *oa_counter)
{
   struct brw_perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = GL_PERFQUERY_COUNTER_THROUGHPUT_INTEL;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;
   counter->offset = pot_align(builder->offset, 8);
   counter->size = sizeof(uint64_t);

   builder->offset = counter->offset + counter->size;
}

static void
report_uint64_oa_counter_as_duration(struct brw_query_builder *builder,
                                     const char *name,
                                     const char *desc,
                                     struct brw_oa_counter *oa_counter)
{
   struct brw_perf_query_counter *counter =
      &builder->query->counters[builder->query->n_counters++];

   counter->oa_counter = oa_counter;
   counter->name = name;
   counter->desc = desc;
   counter->type = GL_PERFQUERY_COUNTER_DURATION_RAW_INTEL;
   counter->data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL;
   counter->raw_max = 0;
   counter->offset = pot_align(builder->offset, 8);
   counter->size = sizeof(uint64_t);

   builder->offset = counter->offset + counter->size;
}

static void
add_pipeline_stage_counters(struct brw_query_builder *builder,
                            const char *short_name,
                            const char *long_name,
                            int aggregate_active_counter,
                            int aggregate_stall_counter,
                            int n_threads_counter)
{
   struct brw_oa_counter *active, *stall, *n_threads, *c;
   char *short_desc;
   char *long_desc;


   short_desc = ralloc_asprintf(builder->brw, "%s EU Active", short_name);
   long_desc = ralloc_asprintf(builder->brw,
                               "The percentage of time in which %s were "
                               "processed actively on the EUs.", long_name);
   active = add_raw_oa_counter(builder, aggregate_active_counter);
   c = add_oa_counter_normalised_by_eu_duration(builder, active);
   report_float_oa_counter_as_percentage_duration(builder, short_desc, long_desc, c);


   short_desc = ralloc_asprintf(builder->brw, "%s EU Stall", short_name);
   long_desc = ralloc_asprintf(builder->brw,
                               "The percentage of time in which %s were "
                               "stalled on the EUs.", long_name);
   stall = add_raw_oa_counter(builder, aggregate_stall_counter);
   c = add_oa_counter_normalised_by_eu_duration(builder, stall);
   report_float_oa_counter_as_percentage_duration(builder, short_desc, long_desc, c);


   n_threads = add_raw_oa_counter(builder, n_threads_counter);

   short_desc = ralloc_asprintf(builder->brw, "%s AVG Active per Thread",
                                short_name);
   long_desc = ralloc_asprintf(builder->brw,
                               "The average number of cycles per hardware "
                               "thread run in which %s were processed actively "
                               "on the EUs.", long_name);
   c = add_average_thread_cycles_oa_counter(builder, active, n_threads);
   report_uint64_oa_counter_as_raw_uint64(builder, short_desc, long_desc, c);


   short_desc = ralloc_asprintf(builder->brw, "%s AVG Stall per Thread",
                                short_name);
   long_desc = ralloc_asprintf(builder->brw,
                               "The average number of cycles per hardware "
                               "thread run in which %s were stalled "
                               "on the EUs.", long_name);
   c = add_average_thread_cycles_oa_counter(builder, stall, n_threads);
   report_uint64_oa_counter_as_raw_uint64(builder, short_desc, long_desc, c);
}

static void
add_aggregate_counters(struct brw_query_builder *builder)
{
   struct brw_oa_counter *raw;
   struct brw_oa_counter *c;
   int a_offset = builder->a_offset;

   raw = add_raw_oa_counter(builder, a_offset + 41);
   c = add_oa_counter_normalised_by_gpu_duration(builder, raw);
   report_float_oa_counter_as_percentage_duration(builder,
                                                  "GPU Busy",
                                                  "The percentage of time in which the GPU has being processing GPU commands.",
                                                  c);

   raw = add_raw_oa_counter(builder, a_offset); /* aggregate EU active */
   c = add_oa_counter_normalised_by_eu_duration(builder, raw);
   report_float_oa_counter_as_percentage_duration(builder,
                                                   "EU Active",
                                                   "The percentage of time in which the Execution Units were actively processing.",
                                                   c);

   raw = add_raw_oa_counter(builder, a_offset + 1); /* aggregate EU stall */
   c = add_oa_counter_normalised_by_eu_duration(builder, raw);
   report_float_oa_counter_as_percentage_duration(builder,
                                                   "EU Stall",
                                                   "The percentage of time in which the Execution Units were stalled.",
                                                   c);

   add_pipeline_stage_counters(builder,
                               "VS",
                               "vertex shaders",
                               a_offset + 2, /* aggregate active */
                               a_offset + 3, /* aggregate stall */
                               a_offset + 5); /* n threads loaded */

   /* Not currently supported by Mesa... */
#if 0
   add_pipeline_stage_counters(builder,
                               "HS",
                               "hull shaders",
                               a_offset + 7, /* aggregate active */
                               a_offset + 8, /* aggregate stall */
                               a_offset + 10); /* n threads loaded */

   add_pipeline_stage_counters(builder,
                               "DS",
                               "domain shaders",
                               a_offset + 12, /* aggregate active */
                               a_offset + 13, /* aggregate stall */
                               a_offset + 15); /* n threads loaded */

   add_pipeline_stage_counters(builder,
                               "CS",
                               "compute shaders",
                               a_offset + 17, /* aggregate active */
                               a_offset + 18, /* aggregate stall */
                               a_offset + 20); /* n threads loaded */
#endif

   add_pipeline_stage_counters(builder,
                               "GS",
                               "geometry shaders",
                               a_offset + 22, /* aggregate active */
                               a_offset + 23, /* aggregate stall */
                               a_offset + 25); /* n threads loaded */

   add_pipeline_stage_counters(builder,
                               "PS",
                               "pixel shaders",
                               a_offset + 27, /* aggregate active */
                               a_offset + 28, /* aggregate stall */
                               a_offset + 30); /* n threads loaded */

   raw = add_raw_oa_counter(builder, a_offset + 32); /* hiz fast z passing */
   raw = add_raw_oa_counter(builder, a_offset + 33); /* hiz fast z failing */

   raw = add_raw_oa_counter(builder, a_offset + 42); /* vs bottleneck */
   raw = add_raw_oa_counter(builder, a_offset + 43); /* gs bottleneck */
}

static void
hsw_add_basic_oa_counter_query(struct brw_context *brw)
{
   struct brw_query_builder builder;
   struct brw_perf_query *query =
      &brw->perfquery.queries[brw->perfquery.n_queries++];
   struct brw_oa_counter *elapsed;
   struct brw_oa_counter *c;
   struct brw_perf_query_counter *last;
   int a_offset = 3; /* A0 */
   int b_offset = a_offset + 45; /* B0 */

   query->kind = OA_COUNTERS;
   query->name = "Gen7 Basic Observability Architecture Counters";
   query->counters = rzalloc_array(brw, struct brw_perf_query_counter,
                                   MAX_PERF_QUERY_COUNTERS);
   query->n_counters = 0;
   query->oa_counters = rzalloc_array(brw, struct brw_oa_counter,
                                      MAX_OA_QUERY_COUNTERS);
   query->n_oa_counters = 0;
   query->perf_profile_id = 0; /* default profile */
   query->perf_oa_format_id = I915_PERF_OA_FORMAT_A45_B8_C8_HSW;

   builder.brw = brw;
   builder.query = query;
   builder.offset = 0;
   builder.next_accumulator_index = 0;

   builder.a_offset = a_offset;
   builder.b_offset = b_offset;
   builder.c_offset = -1;

   /* Can be referenced by other counters... */
   builder.gpu_core_clock = add_raw_oa_counter(&builder, b_offset);

   elapsed = add_hsw_elapsed_oa_counter(&builder);
   report_uint64_oa_counter_as_duration(&builder,
                                        "GPU Time Elapsed",
                                        "Time elapsed on the GPU during the measurement.",
                                        elapsed);

   c = add_avg_frequency_oa_counter(&builder, elapsed);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "AVG GPU Core Frequency",
                                            "Average GPU Core Frequency in the measurement.",
                                            c);

   add_aggregate_counters(&builder);

   assert(query->n_counters < MAX_PERF_QUERY_COUNTERS);
   assert(query->n_oa_counters < MAX_OA_QUERY_COUNTERS);

   last = &query->counters[query->n_counters - 1];
   query->data_size = last->offset + last->size;
}

static void
hsw_add_3d_oa_counter_query(struct brw_context *brw)
{
   struct brw_query_builder builder;
   struct brw_perf_query *query =
      &brw->perfquery.queries[brw->perfquery.n_queries++];
   int a_offset;
   int b_offset;
   int c_offset;
   struct brw_oa_counter *elapsed;
   struct brw_oa_counter *raw;
   struct brw_oa_counter *c;
   struct brw_oa_counter *sampler0_busy_raw;
   struct brw_oa_counter *sampler1_busy_raw;
   struct brw_oa_counter *sampler0_bottleneck;
   struct brw_oa_counter *sampler1_bottleneck;
   struct brw_oa_counter *sampler0_texels;
   struct brw_oa_counter *sampler1_texels;
   struct brw_oa_counter *sampler0_l1_misses;
   struct brw_oa_counter *sampler1_l1_misses;
   struct brw_oa_counter *sampler_l1_misses;
   struct brw_perf_query_counter *last;

   query->kind = OA_COUNTERS;
   query->name = "Gen7 3D Observability Architecture Counters";
   query->counters = rzalloc_array(brw, struct brw_perf_query_counter,
                                   MAX_PERF_QUERY_COUNTERS);
   query->n_counters = 0;
   query->oa_counters = rzalloc_array(brw, struct brw_oa_counter,
                                      MAX_OA_QUERY_COUNTERS);
   query->n_oa_counters = 0;
   query->perf_profile_id = I915_PERF_OA_PROFILE_3D;
   query->perf_oa_format_id = I915_PERF_OA_FORMAT_A45_B8_C8_HSW;

   builder.brw = brw;
   builder.query = query;
   builder.offset = 0;
   builder.next_accumulator_index = 0;

   /* A counters offset = 12  bytes / 0x0c (45 A counters)
    * B counters offset = 192 bytes / 0xc0 (8  B counters)
    * C counters offset = 224 bytes / 0xe0 (8  C counters)
    *
    * Note: we index into the snapshots/reports as arrays of uint32 values
    * relative to the A/B/C offset since different report layouts can vary how
    * many A/B/C counters but with relative addressing it should be possible to
    * re-use code for describing the counters available with different report
    * layouts.
    */

   builder.a_offset = a_offset = 3;
   builder.b_offset = b_offset = a_offset + 45;
   builder.c_offset = c_offset = b_offset + 8;

   /* Can be referenced by other counters... */
   builder.gpu_core_clock = add_raw_oa_counter(&builder, c_offset + 2);

   elapsed = add_hsw_elapsed_oa_counter(&builder);
   report_uint64_oa_counter_as_duration(&builder,
                                        "GPU Time Elapsed",
                                        "Time elapsed on the GPU during the measurement.",
                                        elapsed);

   c = add_avg_frequency_oa_counter(&builder, elapsed);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "AVG GPU Core Frequency",
                                            "Average GPU Core Frequency in the measurement.",
                                            c);

   add_aggregate_counters(&builder);

   raw = add_raw_oa_counter(&builder, a_offset + 35);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Early Depth Test Fails",
                                            "The total number of pixels dropped on early depth test.",
                                            raw);
   /* XXX: caveat: it's 2x real No. when PS has 2 output colors */
   raw = add_raw_oa_counter(&builder, a_offset + 36);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Samples Killed in PS",
                                            "The total number of samples or pixels dropped in pixel shaders.",
                                            raw);
   raw = add_raw_oa_counter(&builder, a_offset + 37);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Alpha Test Fails",
                                            "The total number of pixels dropped on post-PS alpha test.",
                                            raw);
   raw = add_raw_oa_counter(&builder, a_offset + 38);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Late Stencil Test Fails",
                                            "The total number of pixels dropped on post-PS stencil test.",
                                            raw);
   raw = add_raw_oa_counter(&builder, a_offset + 39);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Late Depth Test Fails",
                                            "The total number of pixels dropped on post-PS depth test.",
                                            raw);
   raw = add_raw_oa_counter(&builder, a_offset + 40);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Samples Written",
                                            "The total number of samples or pixels written to all render targets.",
                                            raw);

   raw = add_raw_oa_counter(&builder, c_offset + 5);
   /* I.e. assuming even work distribution across threads... */
   c = add_scaled_uint64_oa_counter(&builder, raw, brw->perfquery.eu_count * 4);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Samples Blended",
                                            "The total number of blended samples or pixels written to all render targets.",
                                            c);

#warning "check GT has slice 0 + 1"
   /* XXX: XML implies explicit sub-slice availability check, but surely we can assume we have a slice 0? */
   sampler0_busy_raw = add_raw_oa_counter(&builder, b_offset + 0);
   c = add_oa_counter_normalised_by_gpu_duration(&builder, sampler0_busy_raw);
   report_float_oa_counter_as_percentage_duration(&builder,
                                                   "Sampler 0 Busy",
                                                   "The percentage of time in which sampler 0 was busy.",
                                                   c);
   /* XXX: XML implies explicit sub-slice availability check: might just have one sampler? */
   sampler1_busy_raw = add_raw_oa_counter(&builder, b_offset + 1);
   c = add_oa_counter_normalised_by_gpu_duration(&builder, sampler1_busy_raw);
   report_float_oa_counter_as_percentage_duration(&builder,
                                                   "Sampler 1 Busy",
                                                   "The percentage of time in which sampler 1 was busy.",
                                                   c);

   c = add_hsw_samplers_busy_duration_oa_counter(&builder,
                                                 sampler0_busy_raw,
                                                 sampler1_busy_raw);
   report_float_oa_counter_as_percentage_duration(&builder,
                                                   "Samplers Busy",
                                                   "The percentage of time in which samplers were busy.",
                                                   c);

   raw = add_raw_oa_counter(&builder, b_offset + 2);
   sampler0_bottleneck = add_oa_counter_normalised_by_gpu_duration(&builder, raw);
   report_float_oa_counter_as_percentage_duration(&builder,
                                                   "Sampler 0 Bottleneck",
                                                   "The percentage of time in which sampler 0 was a bottleneck.",
                                                   sampler0_bottleneck);
   raw = add_raw_oa_counter(&builder, b_offset + 3);
   sampler1_bottleneck = add_oa_counter_normalised_by_gpu_duration(&builder, raw);
   report_float_oa_counter_as_percentage_duration(&builder,
                                                   "Sampler 1 Bottleneck",
                                                   "The percentage of time in which sampler 1 was a bottleneck.",
                                                   sampler1_bottleneck);

   c = add_max_of_float_oa_counters(&builder, sampler0_bottleneck, sampler1_bottleneck);
   report_float_oa_counter_as_percentage_duration(&builder,
                                                   "Sampler Bottleneck",
                                                   "The percentage of time in which samplers were bottlenecks.",
                                                   c);
   raw = add_raw_oa_counter(&builder, b_offset + 4);
   sampler0_texels = add_scaled_uint64_oa_counter(&builder, raw, 4);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Sampler 0 Texels LOD0", /* XXX LODO? */
                                            "The total number of texels lookups in LOD0 in sampler 0 unit.",
                                            sampler0_texels);
   raw = add_raw_oa_counter(&builder, b_offset + 5);
   sampler1_texels = add_scaled_uint64_oa_counter(&builder, raw, 4);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Sampler 1 Texels LOD0", /* XXX LODO? */
                                            "The total number of texels lookups in LOD0 in sampler 1 unit.",
                                            sampler1_texels);

   /* TODO find a test case to try and sanity check the numbers we're getting */
   c = add_hsw_slice_extrapolated_oa_counter(&builder, sampler0_texels, sampler1_texels);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Sampler Texels LOD0",
                                            "The total number of texels lookups in LOD0 in all sampler units.",
                                            c);

   raw = add_raw_oa_counter(&builder, b_offset + 6);
   sampler0_l1_misses = add_scaled_uint64_oa_counter(&builder, raw, 2);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Sampler 0 Cache Misses",
                                            "The total number of misses in L1 sampler caches.",
                                            sampler0_l1_misses);
   raw = add_raw_oa_counter(&builder, b_offset + 7);
   sampler1_l1_misses = add_scaled_uint64_oa_counter(&builder, raw, 2);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Sampler 1 Cache Misses",
                                            "The total number of misses in L1 sampler caches.",
                                            sampler1_l1_misses);
   sampler_l1_misses = add_hsw_slice_extrapolated_oa_counter(&builder, sampler0_l1_misses, sampler1_l1_misses);
   report_uint64_oa_counter_as_uint64_event(&builder,
                                            "Sampler Cache Misses",
                                            "The total number of misses in L1 sampler caches.",
                                            sampler_l1_misses);

   c = add_scaled_uint64_oa_counter(&builder, sampler_l1_misses, 64);
   report_uint64_oa_counter_as_throughput(&builder,
                                          "L3 Sampler Throughput",
                                          "The total number of GPU memory bytes transferred between samplers and L3 caches.",
                                          c);

   raw = add_raw_oa_counter(&builder, c_offset + 1);
   c = add_scaled_uint64_oa_counter(&builder, raw, 64);
   report_uint64_oa_counter_as_throughput(&builder,
                                          "GTI Fixed Pipe Throughput",
                                          "The total number of GPU memory bytes transferred between Fixed Pipeline (Command Dispatch, Input Assembly and Stream Output) and GTI.",
                                          c);

   raw = add_raw_oa_counter(&builder, c_offset + 0);
   c = add_scaled_uint64_oa_counter(&builder, raw, 64);
   report_uint64_oa_counter_as_throughput(&builder,
                                          "GTI Depth Throughput",
                                          "The total number of GPU memory bytes transferred between depth caches and GTI.",
                                          c);
   raw = add_raw_oa_counter(&builder, c_offset + 3);
   c = add_scaled_uint64_oa_counter(&builder, raw, 64);
   report_uint64_oa_counter_as_throughput(&builder,
                                          "GTI RCC Throughput",
                                          "The total number of GPU memory bytes transferred between render color caches and GTI.",
                                          c);
   raw = add_raw_oa_counter(&builder, c_offset + 4);
   c = add_scaled_uint64_oa_counter(&builder, raw, 64);
   report_uint64_oa_counter_as_throughput(&builder,
                                          "GTI L3 Throughput",
                                          "The total number of GPU memory bytes transferred between L3 caches and GTI.",
                                          c);
   raw = add_raw_oa_counter(&builder, c_offset + 6);
   c = add_scaled_uint64_oa_counter(&builder, raw, 128);
   report_uint64_oa_counter_as_throughput(&builder,
                                          "GTI Read Throughput",
                                          "The total number of GPU memory bytes read from GTI.",
                                          c);
   raw = add_raw_oa_counter(&builder, c_offset + 7);
   c = add_scaled_uint64_oa_counter(&builder, raw, 64);
   report_uint64_oa_counter_as_throughput(&builder,
                                          "GTI Write Throughput",
                                          "The total number of GPU memory bytes written to GTI.",
                                          c);

   assert(query->n_counters < MAX_PERF_QUERY_COUNTERS);
   assert(query->n_oa_counters < MAX_OA_QUERY_COUNTERS);

   last = &query->counters[query->n_counters - 1];
   query->data_size = last->offset + last->size;
}


#define NAMED_STAT(REG, NAME, DESC)                         \
   {                                                        \
      .name = NAME,                                         \
      .desc = DESC,                                         \
      .type = GL_PERFQUERY_COUNTER_RAW_INTEL,               \
      .data_type = GL_PERFQUERY_COUNTER_DATA_UINT64_INTEL,  \
      .size = sizeof(uint64_t),                             \
      .pipeline_stat_reg = REG,                             \
   }
#define STAT(REG, DESC) NAMED_STAT(REG, #REG, DESC)

#warning "TODO: rename pipeline statistics"
#warning "TODO: need / 4 for fragment shader invocation count"
static struct brw_perf_query_counter gen6_pipeline_statistics[] = {
   STAT(IA_VERTICES_COUNT,   "N vertices submitted"),
   STAT(IA_PRIMITIVES_COUNT, "N primitives submitted"),
   STAT(VS_INVOCATION_COUNT, "N vertex shader invocations"),
   STAT(GS_INVOCATION_COUNT, "N geometry shader invocations"), /* XXX: check */
   STAT(GS_PRIMITIVES_COUNT, "N geometry shader primitives emitted"),
   STAT(CL_INVOCATION_COUNT, "N primitives entering clipping"),
   STAT(CL_PRIMITIVES_COUNT, "N primitives leaving clipping"),
   STAT(PS_INVOCATION_COUNT, "N fragment shader invocations"), /* XXX: needs / 4 */
   STAT(PS_DEPTH_COUNT,      "N z-pass fragments"),

   NAMED_STAT(GEN6_SO_PRIM_STORAGE_NEEDED, "SO_PRIM_STORAGE_NEEDED",
              "N geometry shader stream-out primitives (total)"),
   NAMED_STAT(GEN6_SO_NUM_PRIMS_WRITTEN,   "SO_NUM_PRIMS_WRITTEN",
              "N geometry shader stream-out primitives (written)"),
};

static struct brw_perf_query_counter gen7_pipeline_statistics[] = {

   STAT(IA_VERTICES_COUNT,   "N vertices submitted"),
   STAT(IA_PRIMITIVES_COUNT, "N primitives submitted"),
   STAT(VS_INVOCATION_COUNT, "N vertex shader invocations"),
   STAT(HS_INVOCATION_COUNT, "N hull shader invocations"),
   STAT(DS_INVOCATION_COUNT, "N domain shader invocations"),
   STAT(GS_INVOCATION_COUNT, "N geometry shader invocations"), /* XXX: check */
   STAT(GS_PRIMITIVES_COUNT, "N geometry shader primitives emitted"),
   STAT(CL_INVOCATION_COUNT, "N primitives entering clipping"),
   STAT(CL_PRIMITIVES_COUNT, "N primitives leaving clipping"),
   STAT(PS_INVOCATION_COUNT, "N fragment shader invocations"), /* XXX: needs / 4 */
   STAT(PS_DEPTH_COUNT,      "N z-pass fragments"),

   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(0), "SO_NUM_PRIMS_WRITTEN (Stream 0)",
              "N stream-out (stream 0) primitives (total)"),
   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(1), "SO_NUM_PRIMS_WRITTEN (Stream 1)",
              "N stream-out (stream 1) primitives (total)"),
   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(2), "SO_NUM_PRIMS_WRITTEN (Stream 2)",
              "N stream-out (stream 2) primitives (total)"),
   NAMED_STAT(GEN7_SO_PRIM_STORAGE_NEEDED(3), "SO_NUM_PRIMS_WRITTEN (Stream 3)",
              "N stream-out (stream 3) primitives (total)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(0), "SO_NUM_PRIMS_WRITTEN (Stream 0)",
              "N stream-out (stream 0) primitives (written)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(1), "SO_NUM_PRIMS_WRITTEN (Stream 1)",
              "N stream-out (stream 1) primitives (written)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(2), "SO_NUM_PRIMS_WRITTEN (Stream 2)",
              "N stream-out (stream 2) primitives (written)"),
   NAMED_STAT(GEN7_SO_NUM_PRIMS_WRITTEN(3), "SO_NUM_PRIMS_WRITTEN (Stream 3)",
              "N stream-out (stream 3) primitives (written)"),
};

#undef STAT
#undef NAMED_STAT

static void
add_pipeline_statistics_query(struct brw_context *brw,
                              const char *name,
                              struct brw_perf_query_counter *counters,
                              int n_counters)
{
   struct brw_perf_query *query =
      &brw->perfquery.queries[brw->perfquery.n_queries++];

   query->kind = PIPELINE_STATS;
   query->name = name;
   query->data_size = sizeof(uint64_t) * n_counters;
   query->n_counters = n_counters;
   query->counters = counters;

   for (int i = 0; i < n_counters; i++) {
      struct brw_perf_query_counter *counter = &counters[i];
      counter->offset = sizeof(uint64_t) * i;
   }
}

void
brw_init_performance_queries(struct brw_context *brw)
{
   struct gl_context *ctx = &brw->ctx;

   ctx->Driver.GetPerfQueryInfo = brw_get_perf_query_info;
   ctx->Driver.GetPerfCounterInfo = brw_get_perf_counter_info;
   ctx->Driver.CreatePerfQuery = brw_create_perf_query;
   ctx->Driver.DeletePerfQuery = brw_delete_perf_query;
   ctx->Driver.BeginPerfQuery = brw_begin_perf_query;
   ctx->Driver.EndPerfQuery = brw_end_perf_query;
   ctx->Driver.WaitPerfQuery = brw_wait_perf_query;
   ctx->Driver.IsPerfQueryReady = brw_is_perf_query_ready;
   ctx->Driver.GetPerfQueryData = brw_get_perf_query_data;

   brw->perfquery.eu_count = get_eu_count(brw->intelScreen->deviceID);

   if (brw->gen == 6) {
      add_pipeline_statistics_query(brw, "Gen6 Pipeline Statistics Registers",
                                    gen6_pipeline_statistics,
                                    (sizeof(gen6_pipeline_statistics)/
                                     sizeof(gen6_pipeline_statistics[0])));
   } else if (brw->gen == 7) {
      add_pipeline_statistics_query(brw, "Gen7 Pipeline Statistics Registers",
                                    gen7_pipeline_statistics,
                                    (sizeof(gen7_pipeline_statistics)/
                                     sizeof(gen7_pipeline_statistics[0])));

      brw->perfquery.read_oa_report_timestamp = hsw_read_report_timestamp;
      hsw_add_basic_oa_counter_query(brw);
      hsw_add_3d_oa_counter_query(brw);
   }

   ctx->PerfQuery.NumQueries = brw->perfquery.n_queries;

   brw->perfquery.unresolved =
      ralloc_array(brw, struct brw_perf_query_object *, 2);
   brw->perfquery.unresolved_elements = 0;
   brw->perfquery.unresolved_array_size = 2;

   brw->perfquery.page_size = sysconf(_SC_PAGE_SIZE);

   brw->perfquery.perf_oa_event_fd = -1;
   brw->perfquery.perf_oa_buffer_size = 1024 * 1024; /* NB: must be power of two */

   brw->perfquery.next_query_start_report_id = 1000;
}
