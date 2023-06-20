#include <getopt.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <roc_shmem.hpp>

using namespace rocshmem;

#define CACHE_LINE_SIZE 64
#define PAGE_SIZE 4096
#define DEF_NUM_THREADS 1
#define DEF_MESSAGE_SIZE 8
#define WINDOW_SIZE 64
#define DEF_NUM_MESSAGES 640000
#define LARGE_MSG_TH 16384
#define DEF_LARGE_NUM_MESSAGES 64000

/* An ROC_SHMEM+threads put message-rate
 * and bandwidth benchmark.
 *
 * Always with 2 processes
 * Thread i on PE 0 sends to thread i on PE 1.
 */

int num_threads;
int num_messages;
int message_size;

int run_bench(int rank, int size);
void print_usage(const char *argv0);

double get_time() {
  double seconds = 0.0;
  struct timespec tv;

  clock_gettime(CLOCK_MONOTONIC, &tv);
  seconds = tv.tv_sec;
  seconds += (double)tv.tv_nsec / 1.0e9;

  return seconds;
}

int run_bench(int rank, int size) {
  int i;
  size_t buffer_size, contig_buffer_size;
  double *t_elapsed;
  double msg_rate, my_msg_rate, bandwidth, my_bandwidth;
  roc_shmem_ctx_t *ctx;
  char *dest_buf, *source_buf;

  num_messages = WINDOW_SIZE * (num_messages / num_threads / WINDOW_SIZE);

  t_elapsed = (double *)calloc(num_threads, sizeof(double));

  /* Allocate array of ctxs */
  ctx = (roc_shmem_ctx_t *)malloc(sizeof(roc_shmem_ctx_t) * num_threads);

  /**
   * Allocate contiguous buffer for all the threads on the target.
   * Ensure that adjacent buffers are not on the same cache line.
   */
  buffer_size = (message_size + CACHE_LINE_SIZE) * sizeof(char);
  contig_buffer_size = buffer_size * num_threads;

  dest_buf = (char *)roc_shmem_malloc(contig_buffer_size);
  memset(dest_buf, 0, sizeof(contig_buffer_size));
  roc_shmem_barrier_all();

  /* Create windows */
  for (i = 0; i < num_threads; i++) {
    int err = roc_shmem_ctx_create(0, &ctx[i]);
    if (err) {
      printf("PE %d: Warning, could not create context %d (%d)\n", rank, i,
             err);
      ctx[i] = ROC_SHMEM_CTX_DEFAULT;
    }
  }

  /* Allocate the source buffers on the device */
  if (rank % 2 == 0) {
    hipMalloc((void **)&source_buf, contig_buffer_size);
  }

#pragma omp parallel
  {
    int tid;
    int win_i, win_post_i, win_posts;
    int my_message_size;
    roc_shmem_ctx_t my_ctx;

    tid = omp_get_thread_num();
    my_message_size = message_size;
    my_ctx = ctx[tid];
    win_posts = num_messages / WINDOW_SIZE;
    if (win_posts * WINDOW_SIZE != num_messages)
      printf(
          "Warning: The final reported numbers will be off. Please choose "
          "number of messages to be a multiple of window size\n");

    if (rank % 2 == 0) {
      /* Putter */
      void *my_source_buf = (void *)&source_buf[tid * buffer_size];
      void *my_dest_buf = (void *)&dest_buf[tid * buffer_size];
      double t_start, t_end;

      /* Warmup */
      for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
        for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
          roc_shmem_ctx_putmem_nbi(my_ctx, my_dest_buf, my_source_buf,
                                   my_message_size, rank + 1);
        }
        roc_shmem_ctx_quiet(my_ctx);
      }

#pragma omp master
      { roc_shmem_barrier_all(); }
#pragma omp barrier

      /* Benchmark */
      t_start = get_time();

      for (win_post_i = 0; win_post_i < win_posts; win_post_i++) {
        for (win_i = 0; win_i < WINDOW_SIZE; win_i++) {
          roc_shmem_ctx_putmem_nbi(my_ctx, my_dest_buf, my_source_buf,
                                   my_message_size, rank + 1);
        }
        roc_shmem_ctx_quiet(my_ctx);
      }

      t_end = get_time();
      t_elapsed[tid] = t_end - t_start;
      ;

    } else {
      /* Target */

      /* Warmup */

#pragma omp master
      { roc_shmem_barrier_all(); }
#pragma omp barrier

      /* Benchmark */
    }
  }

  roc_shmem_barrier_all();

  if (rank % 2 == 0) {
    int thread_i;
    msg_rate = 0;
    bandwidth = 0;
    printf("%-10s\t%-10s\t%-10s\n", "Thread", "Mmsgs/s", "MB/s");
    for (thread_i = 0; thread_i < num_threads; thread_i++) {
      my_msg_rate = ((double)num_messages / t_elapsed[thread_i]) / 1e6;
      my_bandwidth =
          (((double)message_size * (double)num_messages) / (1024 * 1024)) /
          t_elapsed[thread_i];
      printf("%-10d\t%-10.2f\t%-10.2f\n", thread_i, my_msg_rate, my_bandwidth);
      msg_rate += my_msg_rate;
      bandwidth += my_bandwidth;
    }
    printf("\n%-10s\t%-10s\t%-10s\t%-10s\n", "Size", "Threads", "Mmsgs/s",
           "MB/s");
    printf("%-10d\t", message_size);
    printf("%-10d\t", num_threads);
    printf("%f\t", msg_rate);
    printf("%f\n", bandwidth);
  }

  roc_shmem_barrier_all();

  for (i = 0; i < num_threads; i++) roc_shmem_ctx_destroy(ctx[i]);
  free(ctx);
  free(t_elapsed);
  hipFree(source_buf);

  return 0;
}

int main(int argc, char *argv[]) {
  int op, ret;
  int provided, size, rank;

  struct option long_options[] = {
      {.name = "threads", .has_arg = 1, .val = 'T'},
      {.name = "window-size", .has_arg = 1, .val = 'W'},
      {.name = "num-messages", .has_arg = 1, .val = 'M'},
      {.name = "message-size", .has_arg = 1, .val = 'S'},
      {0, 0, 0, 0}};

  num_threads = DEF_NUM_THREADS;
  num_messages = DEF_NUM_MESSAGES;
  message_size = DEF_MESSAGE_SIZE;

  while (1) {
    op = getopt_long(argc, argv, "h?T:W:M:S:w:C:", long_options, NULL);
    if (op == -1) break;

    switch (op) {
      case '?':
      case 'h':
        print_usage(argv[0]);
        return -1;
      case 'T':
        num_threads = atoi(optarg);
        break;
      case 'M':
        num_messages = atoi(optarg);
        break;
      case 'S':
        message_size = atoi(optarg);
        break;
      default:
        printf("Unrecognized argument\n");
        return EXIT_FAILURE;
    }
  }

  if (optind < argc) {
    print_usage(argv[0]);
    return -1;
  }

  if (message_size > LARGE_MSG_TH) {
    if (num_messages == DEF_NUM_MESSAGES) num_messages = DEF_LARGE_NUM_MESSAGES;
  }

  roc_shmem_init();

  size = roc_shmem_n_pes();
  if (size != 2) {
    printf("Run with only two processes.\n");
    roc_shmem_finalize();
  }

  omp_set_num_threads(num_threads);

  rank = roc_shmem_my_pe();

  ret = run_bench(rank, size);
  if (ret) {
    fprintf(stderr, "Error in running bench \n");
    ret = EXIT_FAILURE;
  }

  roc_shmem_finalize();

  return ret;
}

void print_usage(const char *argv0) {
  printf("Usage:\n");
  printf(
      "  mpiexec -n 2 -ppn 1 ###options-to-bind-threads### -hosts "
      "<sender>,<receiver> %s <options>\n",
      argv0);
  printf("\n");
  printf("Options:\n");
  printf("  -T, --threads=<#threads>			number of threads\n");
  printf("  -M, --num-messages=<num_messages>	number of messages\n");
  printf("  -S, --message-size=<message_size>	size of messages\n");
}
