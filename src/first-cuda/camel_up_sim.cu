#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

#define DICE_MIN 1
#define DICE_MAX 3
#define NUM_CAMELS 5
#define FULL_MASK 0xffffffff

__global__ void setup_kernel(curandState *state) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init((unsigned long long)clock() + idx, idx, 0, &state[idx]);
}

template <typename T>
__global__ void camel_up_sim(curandState *state, const int *positions,
                             const bool *remaining_dice, const int *stack,
                             T *results, const T local_runs) {
  int thread_idx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + thread_idx;

  __shared__ T shared_results[NUM_CAMELS];

  if (idx < NUM_CAMELS) {
    shared_results[thread_idx] = 0;
  }
  __syncthreads();

  T thread_results[NUM_CAMELS] = {0};

  // Save the global variables in the local thread
  // so we can reuse them without having to re-read globally.
  int saved_local_positions[NUM_CAMELS];
  bool saved_local_dice[NUM_CAMELS];
  int saved_local_stack[NUM_CAMELS];

  for (int i = 0; i < NUM_CAMELS; i++) {
    saved_local_positions[i] = positions[i];
    saved_local_dice[i] = remaining_dice[i];
    saved_local_stack[i] = stack[i];
  }

  // Instantiate versions of this that can be used within the
  // simulation.
  int local_positions[NUM_CAMELS];
  bool local_dice[NUM_CAMELS];
  int local_stack[NUM_CAMELS];
  bool moved_camels[NUM_CAMELS] = {0, 0, 0, 0, 0};
  int dice_remaining;

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
      local_positions[i] = saved_local_positions[i];
      local_dice[i] = saved_local_dice[i];
      local_stack[i] = saved_local_stack[i];

      if (local_dice[i] == 1) {
        dice_remaining++;
      }
    }

    while (dice_remaining > 0) {

      do {
        camel_to_move = curand(&state[idx]) % NUM_CAMELS;
      } while (!local_dice[camel_to_move]);

      roll = curand(&state[idx]) % DICE_MAX + 1;

      // move that camel and set its dice as rolled.
      local_positions[camel_to_move] += roll;
      local_dice[camel_to_move] = 0;

#pragma unroll
      for (int j = 0; j < NUM_CAMELS; j++) {
        moved_camels[j] = 0;
      }
      moved_camels[camel_to_move] = 1;

      camel_on_top = local_stack[camel_to_move];

      // Move anyone who is on top of the camel that's moving
      while (camel_on_top != -1) {
        local_positions[camel_on_top] += roll;
        moved_camels[camel_on_top] = 1;
        camel_on_top = local_stack[camel_on_top];
      }

#pragma unroll
      for (int i = 0; i < NUM_CAMELS; i++) {
        // If anyone was on the space the stack moved to, make that camel point
        // to the bottom of the new stack
        if ((i != camel_to_move) &&
            (local_positions[i] == local_positions[camel_to_move]) &&
            (local_stack[i] == -1) && (!moved_camels[i])) {
          local_stack[i] = camel_to_move;
        } else if ((local_stack[i] == camel_to_move) &&
                   (local_positions[i] < local_positions[camel_to_move])) {
          // If anyone pointed to camel_to_move and is on a previous space
          // then make them uncovered.
          local_stack[i] = -1;
        }
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

#pragma unroll
  for (int i = 0; i < NUM_CAMELS; i++) {
    for (int offset = 16; offset > 0; offset /= 2) {
      thread_results[i] +=
          __shfl_down_sync(FULL_MASK, thread_results[i], offset);
    }

    // If it's the first thread in a warp - report the result.
    if (threadIdx.x % 32 == 0) {
      atomicAdd(&shared_results[i], thread_results[i]);
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
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
  constexpr int BLOCKS = 24 * 4; // Four per SM on the 4060
  constexpr int THREADS = 256;
  constexpr int RUNS_PER_THREAD = 100000;
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
  int *d_positions;
  bool *d_remainingDice;
  int *d_stack;
  T *d_results;

  curandState *d_state;
  cudaMalloc((void **)&d_state, BLOCKS * THREADS * sizeof(curandState));

  std::cout << "Setting up curand states..." << std::endl;
  setup_kernel<<<BLOCKS, THREADS>>>(d_state);

  std::cout << "Allocating memory on device..." << std::endl;
  cudaMalloc((void **)&d_positions, NUM_CAMELS * sizeof(int));
  cudaMalloc((void **)&d_results, NUM_CAMELS * sizeof(T));
  cudaMalloc((void **)&d_remainingDice, NUM_CAMELS * sizeof(bool));
  cudaMalloc((void **)&d_stack, NUM_CAMELS * sizeof(int));

  cudaMemset(d_results, 0, NUM_CAMELS * sizeof(T));

  std::cout << "Copying to device..." << std::endl;
  cudaMemcpy(d_positions, positions, NUM_CAMELS * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_remainingDice, remainingDice, NUM_CAMELS * sizeof(bool),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_stack, stack, NUM_CAMELS * sizeof(int), cudaMemcpyHostToDevice);

  std::cout << "Starting sim..." << std::endl;
  camel_up_sim<T><<<BLOCKS, THREADS>>>(d_state, d_positions, d_remainingDice,
                                       d_stack, d_results, RUNS_PER_THREAD);

  cudaDeviceSynchronize();

  std::cout << "Copying results back..." << std::endl;
  cudaMemcpy(results, d_results, NUM_CAMELS * sizeof(T),
             cudaMemcpyDeviceToHost);

  std::cout << "Results are:" << std::endl;
  printArray(results, NUM_CAMELS);

  float probs[NUM_CAMELS];
  constexpr float N_float = static_cast<float>(N);
  for (int i = 0; i < NUM_CAMELS; i++) {
    probs[i] = static_cast<float>(results[i]) / N_float;
  }

  std::cout << "Probabilities are..." << std::endl;
  printArray(probs, NUM_CAMELS);

  cudaFree(d_positions);
  cudaFree(d_results);
  cudaFree(d_remainingDice);
  cudaFree(d_state);
  cudaFree(d_stack);

  free(results);
}