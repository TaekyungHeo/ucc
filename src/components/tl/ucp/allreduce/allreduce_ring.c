/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_dt_reduce.h"
#include "components/mc/ucc_mc.h"
#include <stdio.h>

static ucc_rank_t ucc_tl_ucp_allreduce_ring_get_send_block(ucc_subset_t *subset,
                                                           ucc_rank_t trank,
                                                           ucc_rank_t tsize,
                                                           int step)
{
    return ucc_ep_map_eval(subset->map, (trank - step + tsize) % tsize);
}

static ucc_rank_t ucc_tl_ucp_allreduce_ring_get_recv_block(ucc_subset_t *subset,
                                                           ucc_rank_t trank,
                                                           ucc_rank_t tsize,
                                                           int step)
{
    return ucc_ep_map_eval(subset->map, (trank - step - 1 + tsize) % tsize);
}

void print_hex_buffer(const char *tag, void *buf, size_t len, int max_columns) {
    unsigned char *byte_buf = (unsigned char *)buf;
    printf("%s:", tag); // Print the tag followed by a newline
    for (size_t i = 0; i < len; i++) {
        printf("%02x ", byte_buf[i]);
        if ((i + 1) % max_columns == 0) {
            printf("\n");
        }
    }
    if (len % max_columns != 0) {
        printf("\n"); // Ensure printing a newline at the end if not exactly aligned with max_columns
    }
}

void ucc_tl_ucp_allreduce_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args       = &TASK_ARGS(task);
//    ucc_rank_t         size       = task->subset.map.ep_num;
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    ucc_rank_t         trank      = task->subset.myrank;
    ucc_rank_t         tsize      = (ucc_rank_t)task->subset.map.ep_num;
    void              *rbuf       = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  rmem       = TASK_ARGS(task).dst.info.mem_type;
    size_t             count      = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt         = TASK_ARGS(task).dst.info.datatype;
    size_t             data_size  = (count / tsize) * ucc_dt_size(dt);
    ucc_rank_t         sendto, recvfrom, sblock, rblock;
    int                step;
    void              *buf;
//    int    is_avg;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }
    sendto   = ucc_ep_map_eval(task->subset.map, (trank + 1) % tsize);
    recvfrom = ucc_ep_map_eval(task->subset.map, (trank - 1 + tsize) % tsize);
    printf("%s,rank=%d,task=%p,sendto=%d,recvfrom=%d,data_size=%lu\n",
           __func__,
           task->subset.myrank,
           (void *)task,
           sendto,
           recvfrom,
           data_size);

    while (task->tagged.send_posted < tsize - 1) {
        step = task->tagged.send_posted;
        sblock = task->allreduce_ring.get_send_block(&task->subset, trank,
                                                     tsize, step);
        rblock = task->allreduce_ring.get_recv_block(&task->subset, trank,
                                                     tsize, step);
        buf = PTR_OFFSET(rbuf, sblock * data_size);
        printf("%s,rank=%d,", __func__, task->subset.myrank);
        print_hex_buffer("before_send_nb,sblock", buf, data_size, 40);
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(buf, data_size, rmem, sendto, team, task),
            task, out);
        printf("%s,rank=%d,", __func__, task->subset.myrank);
        print_hex_buffer("after_send_nb,sblock", buf, data_size, 40);
        buf = PTR_OFFSET(rbuf, rblock * data_size);
        printf("%s,rank=%d,", __func__, task->subset.myrank);
        print_hex_buffer("before_recv_nb,rblock", buf, data_size, 40);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(buf, data_size, rmem, recvfrom, team, task),
            task, out);
        printf("%s,rank=%d,", __func__, task->subset.myrank);
        print_hex_buffer("after_recv_nb,rblock", buf, data_size, 40);

        if (args->op == UC_OP_SUM) {
          printf("SUM\n");
        } else if {
          printf("SOMETHING ELSE\n");
        }

//       is_avg = (args->op == UCC_OP_AVG) &&
//                 (task->tagged.recv_completed == (size - 1));
//        if (UCC_OK !=
//            (status = ucc_dt_reduce(
//                 r_scratch,
//                 PTR_OFFSET(sbuf, (block_offset + frag_offset) * dt_size),
//                 reduce_target, frag_count, dt, args,
//                 is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0,
//                 AVG_ALPHA(task), task->reduce_scatter_ring.executor,
//                 &task->reduce_scatter_ring.etask))) {
//            tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
//            task->super.status = status;
//            return;
//        }

        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }
    }
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_done", 0);
}

ucc_status_t ucc_tl_ucp_allreduce_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    size_t             count     = TASK_ARGS(task).dst.info.count;
    void              *sbuf      = TASK_ARGS(task).src.info.buffer;
    void              *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  smem      = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t  rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    ucc_rank_t         trank     = task->subset.myrank;
    ucc_rank_t         tsize     = (ucc_rank_t)task->subset.map.ep_num;
    size_t             data_size = (count / tsize) * ucc_dt_size(dt);
    ucc_status_t       status;
    ucc_rank_t         block;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_ring_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    printf("%s,rank=%d,count=%lu,tsize=%d,data_sz=%lu\n",
           __func__,
           task->subset.myrank,
           count, tsize, data_size);

    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        block = task->allreduce_ring.get_send_block(&task->subset, trank, tsize,
                                                    0);
        printf("%s,rank=%d,block=%d\n",
               __func__, task->subset.myrank, block);
        status = ucc_mc_memcpy(PTR_OFFSET(rbuf, data_size * block),
                               sbuf, data_size, rmem, smem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_allreduce_ring_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_sbgp_t *sbgp;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!(task->flags & UCC_TL_UCP_TASK_FLAG_SUBSET)) {
        if (team->cfg.use_reordering) {
            sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
            task->subset.myrank = sbgp->group_rank;
            task->subset.map    = sbgp->map;
        }
    }

    task->allreduce_ring.get_send_block = ucc_tl_ucp_allreduce_ring_get_send_block;
    task->allreduce_ring.get_recv_block = ucc_tl_ucp_allreduce_ring_get_recv_block;
    task->super.post                    = ucc_tl_ucp_allreduce_ring_start;
    task->super.progress                = ucc_tl_ucp_allreduce_ring_progress;

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *     team,
                                            ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    printf("%s,rank=%d,task=%p\n", __func__, task->subset.myrank, (void *)task);
    status = ucc_tl_ucp_allreduce_ring_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}
