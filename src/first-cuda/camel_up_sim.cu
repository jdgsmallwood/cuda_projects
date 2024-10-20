#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#define DICE_MIN 1
#define DICE_MAX 3
#define NUM_CAMELS 5
#define FULL_MASK 0xffffffff

__constant__ int d_positions[NUM_CAMELS];
__constant__ bool d_remaining_dice[NUM_CAMELS];
__constant__ int d_stack[NUM_CAMELS];
__constant__ int local_runs;


__global__ void setup_kernel(curandState *state) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init((unsigned long long)clock() + idx, idx, 0, &state[idx]);
}

template <typename T>
__global__ void camel_up_sim(curandState *state, T *results, const T local_runs) {
  int thread_idx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + thread_idx;
  
  curandState local_state = state[idx];

  __shared__ T shared_results[NUM_CAMELS];

  T thread_results[NUM_CAMELS] = {0};

  // Instantiate versions of this that can be used within the
  // simulation.
  int local_positions[NUM_CAMELS];
  bool local_dice[NUM_CAMELS];
  int local_stack[NUM_CAMELS];
  int dice_remaining;
  int eligible_camels[NUM_CAMELS];
  int camel_to_move;
  int roll;
  int camel_on_top;
  int winner;

  for (int r = 0; r < local_runs; r++) {
    // Begin one simulation
    dice_remaining = 0;

#pragma unroll
    for (int i = 0; i < NUM_CAMELS; i++) {
      // reset local arrays back to saved initial state.
      local_positions[i] = d_positions[i];
      local_dice[i] = d_remaining_dice[i];
      local_stack[i] = d_stack[i];

      if (local_dice[i] == 1) {
        dice_remaining++;
      }
    }

    while (dice_remaining > 0) {
      // Figure out which camel should be moved.
      int j = 0;
    #pragma unroll
      for(int i = 0; i<NUM_CAMELS; i++) {
        if(local_dice[i]) {
            eligible_camels[j] = i;
            j++;
        }
      }
      
      camel_to_move = eligible_camels[curand(&local_state) % dice_remaining];

      // Roll that camel's dice to see how far it moves.
      roll = curand(&local_state) % DICE_MAX + 1;

      // move that camel and set its dice as rolled.
      local_positions[camel_to_move] += roll;
      local_dice[camel_to_move] = 0;

#pragma unroll
      for (int i = 0; i < NUM_CAMELS; i++) {
        // If anyone was on the space the stack moved to, make that camel point
        // to the bottom of the new stack
        if ((i != camel_to_move) &&
            (local_positions[i] == local_positions[camel_to_move]) &&
            (local_stack[i] == -1)) {
          local_stack[i] = camel_to_move;
        } else if ((local_stack[i] == camel_to_move) &&
                   (local_positions[i] < local_positions[camel_to_move])) {
          // If anyone pointed to camel_to_move and is on a previous space
          // then make them uncovered.
          local_stack[i] = -1;
        }
      }

      camel_on_top = local_stack[camel_to_move];

      // Move anyone who is on top of the camel that's moving
      while (camel_on_top != -1) {
        local_positions[camel_on_top] += roll;
        camel_on_top = local_stack[camel_on_top];
      }

      dice_remaining--;
    }

    winner = 0;
#pragma unroll
    for (int i = 1; i < NUM_CAMELS; i++) {
      if (local_positions[i] > local_positions[winner]) {
        winner = i;
      }
    }

    while (local_stack[winner] != -1) {
      winner = local_stack[winner];
    }

    thread_results[winner] += 1;
  }

// Start collecting the results from all the threads.
// Start by shuffling down on a warp basis.
#pragma unroll
  for (int i = 0; i < NUM_CAMELS; i++) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
      thread_results[i] +=
          __shfl_down_sync(FULL_MASK, thread_results[i], offset);
    }

    // If it's the first thread in a warp - report the result to shared memory.
    if (thread_idx % 32 == 0) {
      atomicAdd(&shared_results[i], thread_results[i]);
    }
  }

  __syncthreads();

  // Report block totals back to the global results variable.
  if (thread_idx == 0) {

    
    #pragma unroll
    for (int i = 0; i < NUM_CAMELS; i++) {
    atomicAdd(&results[i], shared_results[i]);
    }
  }
}

template <typename T> void printArray(T arr[], int size) {
  std::cout << "[";
  for (int i = 0; i < size; i++) {
    std::cout << arr[i];
    if (i < size - 1) {
      std::cout << (", ");
    }
  }
  std::cout << "]\n";
}

int main() {

  using T = unsigned long long int;

  std::cout << "Starting program..." << std::endl;
  constexpr int BLOCKS = 24; // Four per SM on the 4060
  constexpr int THREADS = 1024;
  constexpr int RUNS_PER_THREAD = 1000;
  // Without casting one of these to unsigned long long int then this can
  // overflow integer multiplication and return something nonsensical.
  constexpr unsigned long long int N =
      static_cast<unsigned long long int>(BLOCKS) * THREADS * RUNS_PER_THREAD;

  std::cout << "N: " << std::to_string(N) << std::endl;

  std::cout << "Creating host variables..." << std::endl;
  int positions[NUM_CAMELS] = {0, 0, 0, 0, 0};
  bool remainingDice[NUM_CAMELS] = {1, 1, 1, 1, 1};
  int stack[NUM_CAMELS] = {1, 2, 3, 4, -1};
  T *results;
  results = (T *)malloc(NUM_CAMELS * sizeof(T));

  std::cout << "Creating device pointers..." << std::endl;
  T *d_results;

  curandState *d_state;
  cudaMalloc((void **)&d_state, BLOCKS * THREADS * sizeof(curandState));

  std::cout << "Setting up curand states..." << std::endl;
  setup_kernel<<<BLOCKS, THREADS>>>(d_state);

  std::cout << "Allocating memory on device..." << std::endl;
  cudaMalloc((void **)&d_results, NUM_CAMELS * sizeof(T));

  cudaMemset(d_results, 0, NUM_CAMELS * sizeof(T));

  std::cout << "Copying to device..." << std::endl;
  cudaMemcpyToSymbol(d_positions, positions, NUM_CAMELS * sizeof(int));
  cudaMemcpyToSymbol(d_remaining_dice, remainingDice, NUM_CAMELS * sizeof(bool));
  cudaMemcpyToSymbol(d_stack, stack, NUM_CAMELS * sizeof(int));

  std::cout << "Starting sim..." << std::endl;
  camel_up_sim<T><<<BLOCKS, THREADS>>>(d_state, d_results, RUNS_PER_THREAD);

  cudaDeviceSynchronize();

  std::cout << "Copying results back..." << std::endl;
  cudaMemcpy(results, d_results, NUM_CAMELS * sizeof(T),
             cudaMemcpyDeviceToHost);

  std::cout << "Results are:" << std::endl;
  printArray(results, NUM_CAMELS);

  float probs[NUM_CAMELS];
  constexpr float N_float = static_cast<float>(N);
  #pragma unroll
  for (int i = 0; i < NUM_CAMELS; i++) {
    probs[i] = static_cast<float>(results[i]) / N_float;
  }

  std::cout << "Probabilities are..." << std::endl;
  printArray(probs, NUM_CAMELS);

  cudaFree(d_positions);
  cudaFree(d_results);
  cudaFree(d_remaining_dice);
  cudaFree(d_state);
  cudaFree(d_stack);

  free(results);
}