#include "nccl.h"
#include "mpi.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

const int data_size = 128;
const int num_nodes = 16;
const int ppn = 8;
const int num_ranks = num_nodes * ppn;

/*
 * The main communicator is split into groups of 32 ranks, 
 * Every N-th rank of each group are performing RS/AG together (depending on experts_op value)
 * The collective executed by them is controlled by experts_op: 0=RS, 1=AG
 */
int experts_reduction(cudaStream_t stream, ncclComm_t comm, int rank, int size, float *send_buf, float *recv_buf, int experts_op) {

    if (rank == 0){
        std::cout << "Running experts reduction" << std::endl;
    }

    ncclComm_t expertsComm;
    int color = rank % 32;

    ncclCommSplit(comm, color, 0, &expertsComm, NULL);
    switch (experts_op){
        case 0:
            NCCLCHECK(ncclReduceScatter(send_buf, recv_buf, size, ncclFloat, ncclSum, expertsComm, stream));
            break;
        case 1:
            NCCLCHECK(ncclAllGather(send_buf, recv_buf, size/num_ranks, ncclFloat, expertsComm, stream));
            break;
        default:
            std::cerr << "Invalid experts_op value, received "<< experts_op << std::endl;
            return 1;
    }

    //ncclCommDestroy(expertsComm);
    return 0;
}

int pipeline_parallelism(cudaStream_t stream, ncclComm_t comm, int rank, int size, float *send_buf, float *recv_buf) {
    int peer = rank % 32 < 16 ? rank + 16 : rank - 16;

    if (rank == 0){
        std::cout << "Running pipeline parallelism" << peer << std::endl;
    }

    ncclGroupStart();
    NCCLCHECK(ncclSend(send_buf, size, ncclFloat, peer, comm, stream));
    NCCLCHECK(ncclRecv(recv_buf, size, ncclFloat, peer, comm, stream));
    ncclGroupEnd();
    return 0;
}

int experts_parallelism(cudaStream_t stream, ncclComm_t comm, int rank, int size, float *send_buf, float *recv_buf) {

    int count = size / 4;
    int commFirstRank = 16*(rank / 16);
    int peer;

    ncclGroupStart();
    for (int off=0; off < 16; off++) {
        peer = commFirstRank + off;
        std::cout << rank << " - sendrecv to " << peer << ", first rank is " << commFirstRank << std::endl;
        ncclSend(send_buf, count, ncclFloat, peer, comm, stream);
        ncclRecv(recv_buf, count, ncclFloat, peer, comm, stream);
    }
    ncclGroupEnd();
    return 0;
}


int main(int argc, char* argv[]) {
    int rank, size;
    ncclComm_t comm;
    cudaStream_t stream;

    // Initialize buffers
    int send_count = data_size / num_nodes;
    float *send_buf, *recv_buf;
    send_buf = (float *) malloc(sizeof(float) * data_size);
    recv_buf = (float *) malloc(sizeof(float) * data_size);
    for (int i=0; i < data_size; i++){
        send_buf[i] = (float) i;
        recv_buf[i] = (float) 0;
    }

    // Initialize MPI or any other framework to get the rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the rank size matches the expected configuration
    if (size != num_ranks) {
        std::cerr << "Number of ranks doesn't match the expected configuration: "<< size << std::endl;
        MPI_Finalize();
        return -1;
    }

    // Set up NCCL
    cudaStreamCreate(&stream);
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Create communicator 
    NCCLCHECK(ncclCommInitRank(&comm, num_ranks, id, rank));

    // Run desired scenario
    ncclGroupStart();
    //experts_reduction(stream, comm, rank, size, send_buf, recv_buf, 0); 
    experts_parallelism(stream, comm, rank, size, send_buf, recv_buf);
    ncclGroupEnd();

    // Finalize NCCL
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

