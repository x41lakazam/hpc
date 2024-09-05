#include <nccl.h>
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
 * 
 * Ranks 0/32/64/96 do RS or AG
 * Ranks 16/48/80/112 do the same op
 * The collective executed by them is controlled by experts_op: 0=RS, 1=AG
 */
int experts_reduction(cudaStream_t stream, ncclComm_t comm, int rank, int size, ncclFloat *send_buf, ncclFloat *recv_buf, int experts_op) {

    ncclComm_t expertsComm;
    int color = NCCL_SPLIT_NOCOLOR;
    switch (rank % 32){
        case 0: color = 0; break;
        case 1: color = 1; break;
    }

    ncclCommSplit(comm, color, 0, expertsComm, NULL);

    // Experts do AG/RS
    if (color != NCCL_SPLIT_NOCOLOR){
        if (experts_op == 0)
            NCCLCHECK(ncclReduceScatter(send_buf, recv_buf, size, ncclFloat, ncclSum, expertsComm, stream));
        else
            NCCLCHECK(ncclAllGather(send_buf, recv_buf, size/num_ranks, ncclFloat, expertsComm, stream));
    }
    //ncclCommDestroy(expertsComm);
    return 0;
}

int pipeline_parallelism(cudaStream_t stream, ncclComm_t comm, int rank, int size, ncclFloat *send_buf, ncclFloat *recv_buf) {
    //int peer = rank % 32 ? rank + 16 : rank - 16;
    int peer;
    switch (rank){
        case 0: peer = 16; break;
        case 16: peer = 0; break;
        case 32: peer = 48; break;
        case 48: peer = 32; break;
        case 64: peer = 80; break;
        case 80: peer = 64; break;
        case 96: peer = 112; break;
        case 112: peer = 96; break;
        default: return 0;
    }

    ncclGroupStart();
    NCCLCHECK(ncclSend(send_buf, size, ncclFloat, peer, comm, stream));
    NCCLCHECK(ncclRecv(recv_buf, size, ncclFloat, peer, comm, stream));
    ncclGroupEnd();
    return 0;
}

int experts_parallelism(cudaStream_t stream, ncclComm_t comm, int rank, int size, ncclFloat *send_buf, ncclFloat *recv_buf) {

    int count = size / 4;
    int commFirstRank = 16*(rank / 16);

    ncclGroupStart();
    for (int off=0; off < 16; off++) {
        ncclSend(send_buf[off], count, ncclFloat, commFirstRank+off, comm, stream);
        ncclRecv(recvbuff[off], count, ncclFloat, commFirstRank+off, comm, stream);
    }
    ncclGroupEnd();
}


int main(int argc, char* argv[]) {
    int rank, size;
    ncclComm_t comm;
    cudaStream_t stream;

    // Initialize buffers
    int send_count = data_size / num_nodes;
    ncclFloat *send_buf, *recv_buf;
    send_buf = (ncclFloat *) malloc(sizeof(ncclFloat) * data_size);
    recv_buf = (ncclFloat *) malloc(sizeof(ncclFloat) * data_size);
    for (int i=0; i < data_size; i++){
        send_buf[i] = (ncclFloat) i;
        recv_buf[i] = (ncclFloat) 0;
    }

    // Initialize MPI or any other framework to get the rank and size
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the rank size matches the expected configuration
    if (size != num_ranks) {
        std::cerr << "Number of ranks doesn't match the expected configuration!" << std::endl;
        MPI_Finalize();
        return -1;
    }

    // Set up NCCL
    ncclComm_t comm;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Create communicator 
    NCCLCHECK(ncclCommInitRank(&comm, num_ranks, id, rank));

    // Run desired scenario
    ncclGroupStart();
    experts_reduction();
    experts_parallelism();
    ncclGroupEnd();

    // Finalize NCCL
    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

