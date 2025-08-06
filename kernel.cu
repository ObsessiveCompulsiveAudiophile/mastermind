#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>

// CUDA includes
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

// CORE DATA STRUCTURES
const int s = 1296;
const int MASK_WORDS = 21;
const int NUM_MULTIPLIERS = 14;
const int MAX_GAME_LENGTH = 6;

// GA PARAMETERS
const int POPULATION_SIZE = 64;
const int NUM_GENERATIONS = 50000;
const int STAGNATION_THRESHOLD = 2500;
const int MAJOR_RESTART_THRESHOLD = 5000;

// CUDA OPTIMIZATION PARAMETERS
const int THREADS_PER_BLOCK = 256;

// Device constants
__device__ __constant__ int d_POPULATION_SIZE = 64;
__device__ __constant__ int d_NUM_MULTIPLIERS = 14;
__device__ __constant__ int d_MASK_WORDS = 21;
__device__ __constant__ int d_FORCED_FIRST_GUESS = 8;
__device__ __constant__ int d_ELITE_SIZE = 10;
__device__ __constant__ int d_MAX_GAME_LENGTH = 6;
__device__ __constant__ int d_PUNISHMENT_SCORE = 7;

// SoA (Structure of Arrays) layout
struct PopulationSoA {
    float* multipliers_turn2;
    float* multipliers_turn3;
    float* multipliers_turn4;
    float* multipliers_turn5;
    float* multipliers_turn6;
    float* fitness;
    char* is_seeded;
};

struct GameState {
    uint64_t mask[MASK_WORDS];
    int previous_guesses[MAX_GAME_LENGTH];
    int num_previous_guesses;
};

// Forward declarations
std::array<int, s> generateValids();
int calculateFeedbackIndex(int guess_code, int secret_code);
void precomputeBitmasks(std::vector<uint64_t>& masks, const std::array<int, s>& Valids);
void allocatePopulationSoA(PopulationSoA& pop);
void freePopulationSoA(PopulationSoA& pop);
float calculateBaseline(PopulationSoA& d_population, const std::vector<uint64_t>& h_partitionMasks, const std::array<int, s>& Valids);
struct Individual;
void copyFromSoA(std::vector<Individual>& individuals, const PopulationSoA& pop);
void copyToSoA(const std::vector<Individual>& individuals, PopulationSoA& pop);


// Device utility functions
__device__ __forceinline__ bool isBitSet(const GameState& state, int bit_pos) {
    if (bit_pos < 0 || bit_pos >= s) return false;
    return (state.mask[bit_pos >> 6] & (1ULL << (bit_pos & 63))) != 0;
}

__device__ __forceinline__ int intersectAndPopcount(const GameState& state, const uint64_t* maskB_ptr) {
    int count = 0;
#pragma unroll
    for (int i = 0; i < d_MASK_WORDS; ++i) count += __popcll(state.mask[i] & maskB_ptr[i]);
    return count;
}

__device__ __forceinline__ int getTotalValidCodes(const GameState& state) {
    int count = 0;
#pragma unroll
    for (int i = 0; i < d_MASK_WORDS; ++i) count += __popcll(state.mask[i]);
    return count;
}

__device__ __forceinline__ void initializeGameState(GameState& state) {
#pragma unroll
    for (int i = 0; i < d_MASK_WORDS; ++i) state.mask[i] = 0xFFFFFFFFFFFFFFFFULL;
    int remainder = s & 63;
    if (remainder > 0) state.mask[d_MASK_WORDS - 1] = (1ULL << remainder) - 1;
    state.num_previous_guesses = 0;
}

__device__ __forceinline__ void applyFeedbackMask(GameState& state, const uint64_t* feedback_mask) {
#pragma unroll
    for (int i = 0; i < d_MASK_WORDS; ++i) state.mask[i] &= feedback_mask[i];
}

__device__ __forceinline__ void addPreviousGuess(GameState& state, int guess) {
    if (state.num_previous_guesses < d_MAX_GAME_LENGTH) state.previous_guesses[state.num_previous_guesses++] = guess;
}

__device__ __forceinline__ bool isPreviousGuess(const GameState& state, int guess) {
    for (int i = 0; i < state.num_previous_guesses; i++) if (state.previous_guesses[i] == guess) return true;
    return false;
}

__device__ int calculateFeedbackIndexDevice(int guess_idx, int secret_idx, int* valid_codes) {
    int guess_code = valid_codes[guess_idx];
    int secret_code = valid_codes[secret_idx];
    if (guess_code == secret_code) return 13;
    int p = 0, m = 0;
    int guess_digits[4], secret_digits[4];
    int temp_guess = guess_code, temp_secret = secret_code;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        guess_digits[i] = temp_guess % 10; secret_digits[i] = temp_secret % 10;
        temp_guess /= 10; temp_secret /= 10;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) if (guess_digits[i] == secret_digits[i]) { p++; guess_digits[i] = 0; secret_digits[i] = -1; }
    for (int i = 0; i < 4; ++i) if (guess_digits[i] != 0) for (int j = 0; j < 4; ++j) if (guess_digits[i] == secret_digits[j]) { m++; secret_digits[j] = -1; break; }
    return __float2int_rn(-0.5f * p * p + 5.5f * p + m);
}

__device__ __forceinline__ float calculateWeightedShannonFromProbs(float* probabilities, const float* multipliers) {
    float shannon = 0.0f;
    for (int i = 0; i < 14; i++) if (probabilities[i] > 1e-9f) shannon -= multipliers[i] * probabilities[i] * __log2f(probabilities[i]);
    return shannon;
}

__device__ int findBestGuessOptimized(const GameState& state, const uint64_t* partition_masks, const float* multipliers) {
    int total_valid = getTotalValidCodes(state);
    if (total_valid <= 1) {
        for (int i = 0; i < s; i++) if (isBitSet(state, i) && !isPreviousGuess(state, i)) return i;
        return -1;
    }

    float scores[s];
    for (int guess_idx = 0; guess_idx < s; guess_idx++) {
        if (isPreviousGuess(state, guess_idx)) {
            scores[guess_idx] = -1e9f; continue;
        }

        int partition_counts[14];
        for (int feedback = 0; feedback < 14; feedback++) {
            const uint64_t* p_mask = &partition_masks[(guess_idx * 14 + feedback) * MASK_WORDS];
            partition_counts[feedback] = intersectAndPopcount(state, p_mask);
        }

        float probabilities[14];
        float inv_total = __fdividef(1.0f, (float)total_valid);
        for (int f = 0; f < 14; f++) probabilities[f] = (float)partition_counts[f] * inv_total;

        scores[guess_idx] = calculateWeightedShannonFromProbs(probabilities, multipliers);
    }

    float max_score = -1e8f;
    for (int i = 0; i < s; i++) if (scores[i] > max_score) max_score = scores[i];

    for (int i = 0; i < s; i++) if (fabsf(scores[i] - max_score) < 1e-6f && isBitSet(state, i)) return i;
    for (int i = 0; i < s; i++) if (fabsf(scores[i] - max_score) < 1e-6f) return i;

    return -1;
}

__device__ int playCompleteGameOptimized(int secret_idx, const PopulationSoA& pop, int individual_idx, const uint64_t* partition_masks, int* valid_codes, bool is_baseline_run) {
    GameState state;
    initializeGameState(state);
    int turn = 1;
    int current_guess = d_FORCED_FIRST_GUESS;
    while (turn <= d_MAX_GAME_LENGTH) {
        addPreviousGuess(state, current_guess);
        int feedback_idx = calculateFeedbackIndexDevice(current_guess, secret_idx, valid_codes);
        if (feedback_idx == 13) return turn;
        const uint64_t* feedback_mask = &partition_masks[(current_guess * 14 + feedback_idx) * d_MASK_WORDS];
        applyFeedbackMask(state, feedback_mask);
        turn++;
        if (turn > d_MAX_GAME_LENGTH) return d_PUNISHMENT_SCORE;

        float multipliers[14];
        if (is_baseline_run) {
            for (int i = 0; i < 14; i++) multipliers[i] = 1.0f;
        }
        else {
            switch (turn) {
            case 2: for (int i = 0; i < 14; i++) multipliers[i] = pop.multipliers_turn2[individual_idx * d_NUM_MULTIPLIERS + i]; break;
            case 3: for (int i = 0; i < 14; i++) multipliers[i] = pop.multipliers_turn3[individual_idx * d_NUM_MULTIPLIERS + i]; break;
            case 4: for (int i = 0; i < 14; i++) multipliers[i] = pop.multipliers_turn4[individual_idx * d_NUM_MULTIPLIERS + i]; break;
            case 5: for (int i = 0; i < 14; i++) multipliers[i] = pop.multipliers_turn5[individual_idx * d_NUM_MULTIPLIERS + i]; break;
            case 6: for (int i = 0; i < 14; i++) multipliers[i] = pop.multipliers_turn6[individual_idx * d_NUM_MULTIPLIERS + i]; break;
            default: return d_PUNISHMENT_SCORE;
            }
        }
        current_guess = findBestGuessOptimized(state, partition_masks, multipliers);
        if (current_guess == -1) return d_PUNISHMENT_SCORE;
    }
    return d_PUNISHMENT_SCORE;
}

__global__ void evaluateFullGameOptimized(PopulationSoA pop, uint64_t* partition_masks, int* valid_codes, bool is_baseline_run) {
    int individual_idx = blockIdx.x;
    int secret_idx = blockIdx.y * blockDim.x + threadIdx.x;

    int num_individuals = is_baseline_run ? 1 : d_POPULATION_SIZE;
    if (individual_idx >= num_individuals || secret_idx >= s) return;

    int game_length = playCompleteGameOptimized(secret_idx, pop, individual_idx, partition_masks, valid_codes, is_baseline_run);
    atomicAdd(&pop.fitness[individual_idx], (float)game_length);
}

__global__ void adaptiveGAOperationsOptimized(PopulationSoA pop, PopulationSoA new_pop, curandState* state, float mutation_rate, bool go_wild, bool major_restart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_POPULATION_SIZE || idx < d_ELITE_SIZE) return;
    curandState localState = state[idx];
    int parent1_idx = __float2int_rn(curand_uniform(&localState) * d_POPULATION_SIZE);
    int parent2_idx = __float2int_rn(curand_uniform(&localState) * d_POPULATION_SIZE);
    while (parent1_idx == parent2_idx) parent2_idx = __float2int_rn(curand_uniform(&localState) * d_POPULATION_SIZE);
    float blend_factor = curand_uniform(&localState);
    for (int i = 0; i < d_NUM_MULTIPLIERS; i++) {
        int offset = idx * d_NUM_MULTIPLIERS + i;
        new_pop.multipliers_turn2[offset] = blend_factor * pop.multipliers_turn2[parent1_idx * d_NUM_MULTIPLIERS + i] + (1.0f - blend_factor) * pop.multipliers_turn2[parent2_idx * d_NUM_MULTIPLIERS + i];
        new_pop.multipliers_turn3[offset] = blend_factor * pop.multipliers_turn3[parent1_idx * d_NUM_MULTIPLIERS + i] + (1.0f - blend_factor) * pop.multipliers_turn3[parent2_idx * d_NUM_MULTIPLIERS + i];
        new_pop.multipliers_turn4[offset] = blend_factor * pop.multipliers_turn4[parent1_idx * d_NUM_MULTIPLIERS + i] + (1.0f - blend_factor) * pop.multipliers_turn4[parent2_idx * d_NUM_MULTIPLIERS + i];
        new_pop.multipliers_turn5[offset] = blend_factor * pop.multipliers_turn5[parent1_idx * d_NUM_MULTIPLIERS + i] + (1.0f - blend_factor) * pop.multipliers_turn5[parent2_idx * d_NUM_MULTIPLIERS + i];
        new_pop.multipliers_turn6[offset] = blend_factor * pop.multipliers_turn6[parent1_idx * d_NUM_MULTIPLIERS + i] + (1.0f - blend_factor) * pop.multipliers_turn6[parent2_idx * d_NUM_MULTIPLIERS + i];
    }
    float base_mutation_rate = go_wild ? mutation_rate * 3.0f : mutation_rate;
    for (int i = 0; i < d_NUM_MULTIPLIERS; i++) {
        int offset = idx * d_NUM_MULTIPLIERS + i;
        if (curand_uniform(&localState) < base_mutation_rate) { new_pop.multipliers_turn2[offset] += curand_normal(&localState) * 0.1f; new_pop.multipliers_turn2[offset] = fmaxf(0.001f, new_pop.multipliers_turn2[offset]); }
        if (curand_uniform(&localState) < base_mutation_rate) { new_pop.multipliers_turn3[offset] += curand_normal(&localState) * 0.1f; new_pop.multipliers_turn3[offset] = fmaxf(0.001f, new_pop.multipliers_turn3[offset]); }
        if (curand_uniform(&localState) < base_mutation_rate) { new_pop.multipliers_turn4[offset] += curand_normal(&localState) * 0.1f; new_pop.multipliers_turn4[offset] = fmaxf(0.001f, new_pop.multipliers_turn4[offset]); }
        if (curand_uniform(&localState) < base_mutation_rate) { new_pop.multipliers_turn5[offset] += curand_normal(&localState) * 0.1f; new_pop.multipliers_turn5[offset] = fmaxf(0.001f, new_pop.multipliers_turn5[offset]); }
        if (curand_uniform(&localState) < base_mutation_rate) { new_pop.multipliers_turn6[offset] += curand_normal(&localState) * 0.1f; new_pop.multipliers_turn6[offset] = fmaxf(0.001f, new_pop.multipliers_turn6[offset]); }
    }
    if (major_restart && curand_uniform(&localState) < 0.3f) {
        for (int i = 0; i < d_NUM_MULTIPLIERS; i++) {
            int offset = idx * d_NUM_MULTIPLIERS + i;
            new_pop.multipliers_turn2[offset] = curand_uniform(&localState); new_pop.multipliers_turn3[offset] = curand_uniform(&localState);
            new_pop.multipliers_turn4[offset] = curand_uniform(&localState); new_pop.multipliers_turn5[offset] = curand_uniform(&localState);
            new_pop.multipliers_turn6[offset] = curand_uniform(&localState);
        }
    }
    new_pop.fitness[idx] = 0.0f;
    new_pop.is_seeded[idx] = 0;
    state[idx] = localState;
}

__global__ void initPopulationOptimized(PopulationSoA pop, curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_POPULATION_SIZE) return;
    curandState localState = state[idx];
    for (int i = 0; i < d_NUM_MULTIPLIERS; i++) {
        int offset = idx * d_NUM_MULTIPLIERS + i;
        pop.multipliers_turn2[offset] = curand_uniform(&localState); pop.multipliers_turn3[offset] = curand_uniform(&localState);
        pop.multipliers_turn4[offset] = curand_uniform(&localState); pop.multipliers_turn5[offset] = curand_uniform(&localState);
        pop.multipliers_turn6[offset] = curand_uniform(&localState);
    }
    pop.fitness[idx] = 0.0f;
    pop.is_seeded[idx] = 0;
    state[idx] = localState;
}

__global__ void initRandom(curandState* state, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_POPULATION_SIZE) curand_init(seed + idx, idx, 0, &state[idx]);
}

void allocatePopulationSoA(PopulationSoA& pop) {
    cudaMalloc(&pop.multipliers_turn2, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float)); cudaMalloc(&pop.multipliers_turn3, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float));
    cudaMalloc(&pop.multipliers_turn4, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float)); cudaMalloc(&pop.multipliers_turn5, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float));
    cudaMalloc(&pop.multipliers_turn6, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float)); cudaMalloc(&pop.fitness, POPULATION_SIZE * sizeof(float));
    cudaMalloc(&pop.is_seeded, POPULATION_SIZE * sizeof(char));
}

void freePopulationSoA(PopulationSoA& pop) {
    cudaFree(pop.multipliers_turn2); cudaFree(pop.multipliers_turn3); cudaFree(pop.multipliers_turn4);
    cudaFree(pop.multipliers_turn5); cudaFree(pop.multipliers_turn6); cudaFree(pop.fitness); cudaFree(pop.is_seeded);
}

struct Individual {
    float multipliers_turn2[NUM_MULTIPLIERS], multipliers_turn3[NUM_MULTIPLIERS], multipliers_turn4[NUM_MULTIPLIERS], multipliers_turn5[NUM_MULTIPLIERS], multipliers_turn6[NUM_MULTIPLIERS];
    float fitness; bool is_seeded;
};

void copyFromSoA(std::vector<Individual>& individuals, const PopulationSoA& pop) {
    std::vector<float> h_mult2(POPULATION_SIZE * NUM_MULTIPLIERS), h_mult3(POPULATION_SIZE * NUM_MULTIPLIERS), h_mult4(POPULATION_SIZE * NUM_MULTIPLIERS), h_mult5(POPULATION_SIZE * NUM_MULTIPLIERS), h_mult6(POPULATION_SIZE * NUM_MULTIPLIERS);
    std::vector<float> h_fitness(POPULATION_SIZE); std::vector<char> h_seeded(POPULATION_SIZE);
    cudaMemcpy(h_mult2.data(), pop.multipliers_turn2, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mult3.data(), pop.multipliers_turn3, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mult4.data(), pop.multipliers_turn4, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mult5.data(), pop.multipliers_turn5, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mult6.data(), pop.multipliers_turn6, POPULATION_SIZE * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fitness.data(), pop.fitness, POPULATION_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_seeded.data(), pop.is_seeded, POPULATION_SIZE * sizeof(char), cudaMemcpyDeviceToHost);
    for (int i = 0; i < POPULATION_SIZE; i++) {
        individuals[i].fitness = h_fitness[i]; individuals[i].is_seeded = (h_seeded[i] != 0);
        for (int j = 0; j < NUM_MULTIPLIERS; j++) {
            individuals[i].multipliers_turn2[j] = h_mult2[i * NUM_MULTIPLIERS + j]; individuals[i].multipliers_turn3[j] = h_mult3[i * NUM_MULTIPLIERS + j];
            individuals[i].multipliers_turn4[j] = h_mult4[i * NUM_MULTIPLIERS + j]; individuals[i].multipliers_turn5[j] = h_mult5[i * NUM_MULTIPLIERS + j];
            individuals[i].multipliers_turn6[j] = h_mult6[i * NUM_MULTIPLIERS + j];
        }
    }
}

void copyToSoA(const std::vector<Individual>& individuals, PopulationSoA& pop) {
    std::vector<float> h_mult2(POPULATION_SIZE * NUM_MULTIPLIERS), h_mult3(POPULATION_SIZE * NUM_MULTIPLIERS), h_mult4(POPULATION_SIZE * NUM_MULTIPLIERS), h_mult5(POPULATION_SIZE * NUM_MULTIPLIERS), h_mult6(POPULATION_SIZE * NUM_MULTIPLIERS);
    std::vector<float> h_fitness(POPULATION_SIZE); std::vector<char> h_seeded(POPULATION_SIZE);
    for (int i = 0; i < individuals.size(); i++) {
        h_fitness[i] = individuals[i].fitness; h_seeded[i] = individuals[i].is_seeded ? 1 : 0;
        for (int j = 0; j < NUM_MULTIPLIERS; j++) {
            h_mult2[i * NUM_MULTIPLIERS + j] = individuals[i].multipliers_turn2[j]; h_mult3[i * NUM_MULTIPLIERS + j] = individuals[i].multipliers_turn3[j];
            h_mult4[i * NUM_MULTIPLIERS + j] = individuals[i].multipliers_turn4[j]; h_mult5[i * NUM_MULTIPLIERS + j] = individuals[i].multipliers_turn5[j];
            h_mult6[i * NUM_MULTIPLIERS + j] = individuals[i].multipliers_turn6[j];
        }
    }
    cudaMemcpy(pop.multipliers_turn2, h_mult2.data(), individuals.size() * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pop.multipliers_turn3, h_mult3.data(), individuals.size() * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pop.multipliers_turn4, h_mult4.data(), individuals.size() * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pop.multipliers_turn5, h_mult5.data(), individuals.size() * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pop.multipliers_turn6, h_mult6.data(), individuals.size() * NUM_MULTIPLIERS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pop.fitness, h_fitness.data(), individuals.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(pop.is_seeded, h_seeded.data(), individuals.size() * sizeof(char), cudaMemcpyHostToDevice);
}

float calculateBaseline(PopulationSoA& d_population, const std::vector<uint64_t>& h_partitionMasks, const std::array<int, s>& Valids) {
    std::cout << "=== CALCULATING BASELINE (all weights = 1.0) ===\n";
    auto baseline_start = std::chrono::high_resolution_clock::now();

    std::vector<Individual> baseline_pop(1);
    for (int i = 0; i < NUM_MULTIPLIERS; ++i) {
        baseline_pop[0].multipliers_turn2[i] = 1.0f; baseline_pop[0].multipliers_turn3[i] = 1.0f;
        baseline_pop[0].multipliers_turn4[i] = 1.0f; baseline_pop[0].multipliers_turn5[i] = 1.0f;
        baseline_pop[0].multipliers_turn6[i] = 1.0f;
    }

    copyToSoA(baseline_pop, d_population);

    uint64_t* d_partition_masks;
    int* d_valid_codes;
    cudaMalloc(&d_partition_masks, h_partitionMasks.size() * sizeof(uint64_t));
    cudaMalloc(&d_valid_codes, s * sizeof(int));
    std::vector<int> h_valid_codes(s);
    for (int i = 0; i < s; i++) h_valid_codes[i] = Valids[i];
    cudaMemcpy(d_partition_masks, h_partitionMasks.data(), h_partitionMasks.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid_codes, h_valid_codes.data(), s * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_population.fitness, 0, sizeof(float));
    dim3 baseline_gridSize(1, (s + 32 - 1) / 32);
    dim3 baseline_blockSize(32);
    evaluateFullGameOptimized << <baseline_gridSize, baseline_blockSize >> > (d_population, d_partition_masks, d_valid_codes, true);
    cudaDeviceSynchronize();

    float baseline_total;
    cudaMemcpy(&baseline_total, d_population.fitness, sizeof(float), cudaMemcpyDeviceToHost);
    float baseline_avg = baseline_total / s;

    auto baseline_end = std::chrono::high_resolution_clock::now();
    auto baseline_duration = std::chrono::duration_cast<std::chrono::milliseconds>(baseline_end - baseline_start).count();
    std::cout << "Baseline calculation completed in " << baseline_duration << "ms\n";
    std::cout << "Baseline average game length: " << std::fixed << std::setprecision(4) << baseline_avg << " turns\n";
    std::cout << "Expected baseline: 4.3735 turns for standard Shannon entropy\n\n";

    cudaFree(d_partition_masks);
    cudaFree(d_valid_codes);
    return baseline_avg;
}

void runFullGameOptimization() {
    cudaSetDevice(0);
    auto Valids = generateValids();
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "=== MASTERMIND WEIGHTED SHANNON ENTROPY OPTIMIZATION ===\n";
    std::cout << "Optimizing " << (5 * NUM_MULTIPLIERS) << " weight parameters\n\n";
    std::vector<uint64_t> h_partitionMasks(s * 14 * MASK_WORDS, 0);
    precomputeBitmasks(h_partitionMasks, Valids);

    PopulationSoA d_population, d_new_population;
    allocatePopulationSoA(d_population);
    allocatePopulationSoA(d_new_population);

    float baseline_avg = calculateBaseline(d_population, h_partitionMasks, Valids);

    curandState* d_state;
    uint64_t* d_partition_masks;
    int* d_valid_codes;
    cudaMalloc(&d_state, POPULATION_SIZE * sizeof(curandState));
    cudaMalloc(&d_partition_masks, h_partitionMasks.size() * sizeof(uint64_t));
    cudaMalloc(&d_valid_codes, s * sizeof(int));
    std::vector<int> h_valid_codes(s);
    for (int i = 0; i < s; i++) h_valid_codes[i] = Valids[i];
    cudaMemcpy(d_partition_masks, h_partitionMasks.data(), h_partitionMasks.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid_codes, h_valid_codes.data(), s * sizeof(int), cudaMemcpyHostToDevice);

    dim3 init_blockSize(THREADS_PER_BLOCK);
    dim3 init_gridSize((POPULATION_SIZE + init_blockSize.x - 1) / init_blockSize.x);
    dim3 eval_blockSize(32);
    dim3 eval_gridSize(POPULATION_SIZE, (s + eval_blockSize.x - 1) / eval_blockSize.x);

    initRandom << <init_gridSize, init_blockSize >> > (d_state, static_cast<unsigned long long>(time(nullptr)));
    initPopulationOptimized << <init_gridSize, init_blockSize >> > (d_population, d_state);

    std::cout << "Starting optimized evaluation with 2D grid (" << eval_gridSize.x << " x " << eval_gridSize.y << ")...\n\n";
    Individual best_ever;
    best_ever.fitness = -1000.0f;
    float best_avg_turns = 1000.0f;
    int stagnation_count = 0, major_stagnation_count = 0;
    for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
        auto gen_start = std::chrono::high_resolution_clock::now();
        cudaMemset(d_population.fitness, 0, POPULATION_SIZE * sizeof(float));
        evaluateFullGameOptimized << <eval_gridSize, eval_blockSize >> > (d_population, d_partition_masks, d_valid_codes, false);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; break; }
        std::vector<float> h_fitness(POPULATION_SIZE);
        cudaMemcpy(h_fitness.data(), d_population.fitness, POPULATION_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < POPULATION_SIZE; i++) h_fitness[i] = -h_fitness[i] / s;
        cudaMemcpy(d_population.fitness, h_fitness.data(), POPULATION_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        std::vector<Individual> h_population(POPULATION_SIZE);
        copyFromSoA(h_population, d_population);
        std::sort(h_population.begin(), h_population.end(), [](const Individual& a, const Individual& b) { return a.fitness > b.fitness; });
        float current_avg_turns = -h_population[0].fitness;
        bool new_best = false;
        if (current_avg_turns < best_avg_turns) {
            best_ever = h_population[0]; best_avg_turns = current_avg_turns; new_best = true;
            stagnation_count = 0; major_stagnation_count = 0;
        }
        else { stagnation_count++; major_stagnation_count++; }
        bool go_wild = stagnation_count >= STAGNATION_THRESHOLD;
        bool major_restart = major_stagnation_count >= MAJOR_RESTART_THRESHOLD;
        if (major_restart) major_stagnation_count = 0;
        auto gen_end = std::chrono::high_resolution_clock::now();
        auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
        if (new_best || generation % 25 == 0) {
            std::cout << "Gen " << std::setw(4) << generation + 1 << " (" << gen_duration << "ms) - Current: " << std::fixed << std::setprecision(4)
                << current_avg_turns << " turns (Best: " << best_avg_turns << " )";
            if (major_restart) std::cout << " [MAJOR RESTART]"; else if (go_wild) std::cout << " [WILD]";
            if (new_best) {
                std::cout << " >>> NEW BEST!\n\a";
                std::cout << "    T2 weights: {"; for (int i = 0; i < NUM_MULTIPLIERS; ++i) { std::cout << std::fixed << std::setprecision(4) << best_ever.multipliers_turn2[i]; if (i < NUM_MULTIPLIERS - 1) std::cout << ","; } std::cout << "}\n";
                std::cout << "    T3 weights: {"; for (int i = 0; i < NUM_MULTIPLIERS; ++i) { std::cout << std::fixed << std::setprecision(4) << best_ever.multipliers_turn3[i]; if (i < NUM_MULTIPLIERS - 1) std::cout << ","; } std::cout << "}\n";
                std::cout << "    T4 weights: {"; for (int i = 0; i < NUM_MULTIPLIERS; ++i) { std::cout << std::fixed << std::setprecision(4) << best_ever.multipliers_turn4[i]; if (i < NUM_MULTIPLIERS - 1) std::cout << ","; } std::cout << "}\n";
                std::cout << "    T5 weights: {"; for (int i = 0; i < NUM_MULTIPLIERS; ++i) { std::cout << std::fixed << std::setprecision(4) << best_ever.multipliers_turn5[i]; if (i < NUM_MULTIPLIERS - 1) std::cout << ","; } std::cout << "}\n";
                std::cout << "    T6 weights: {"; for (int i = 0; i < NUM_MULTIPLIERS; ++i) { std::cout << std::fixed << std::setprecision(4) << best_ever.multipliers_turn6[i]; if (i < NUM_MULTIPLIERS - 1) std::cout << ","; } std::cout << "}\n";
            }
            std::cout << "\n";
        }
        copyToSoA(h_population, d_new_population);
        float mutation_rate = 0.15f;
        adaptiveGAOperationsOptimized << <init_gridSize, init_blockSize >> > (d_population, d_new_population, d_state, mutation_rate, go_wild, major_restart);
        cudaDeviceSynchronize();
        std::swap(d_population, d_new_population);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "\n=== OPTIMIZATION COMPLETE (" << total_duration << " seconds) ===\n";
    std::cout << "Best average game length: " << std::fixed << std::setprecision(4) << best_avg_turns << " turns\n";
    freePopulationSoA(d_population); freePopulationSoA(d_new_population);
    cudaFree(d_state); cudaFree(d_partition_masks); cudaFree(d_valid_codes);
}

int main() { runFullGameOptimization(); return 0; }

std::array<int, s> generateValids() {
    std::array<int, s> valids_array{}; int index = 0;
    for (int d1 = 1; d1 <= 6; ++d1) for (int d2 = 1; d2 <= 6; ++d2) for (int d3 = 1; d3 <= 6; ++d3) for (int d4 = 1; d4 <= 6; ++d4)
        valids_array[index++] = d1 * 1000 + d2 * 100 + d3 * 10 + d4;
    return valids_array;
}

int calculateFeedbackIndex(int guess_code, int secret_code) {
    if (guess_code == secret_code) return 13;
    int p = 0, m = 0;
    int guess_digits[4], secret_digits[4];
    for (int i = 0; i < 4; ++i) { guess_digits[i] = guess_code % 10; secret_digits[i] = secret_code % 10; guess_code /= 10; secret_code /= 10; }
    for (int i = 0; i < 4; ++i) if (guess_digits[i] == secret_digits[i]) { p++; guess_digits[i] = 0; secret_digits[i] = -1; }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) if (guess_digits[i] != 0 && guess_digits[i] == secret_digits[j]) { m++; secret_digits[j] = -1; break; }
    return static_cast<int>(-0.5f * p * p + 5.5f * p + m);
}

void precomputeBitmasks(std::vector<uint64_t>& masks, const std::array<int, s>& Valids) {
    for (int guess_idx = 0; guess_idx < s; ++guess_idx) for (int secret_idx = 0; secret_idx < s; ++secret_idx) {
        int feedback = calculateFeedbackIndex(Valids[guess_idx], Valids[secret_idx]);
        int word_idx = secret_idx / 64, bit_idx = secret_idx % 64;
        uint64_t* targetMask = &masks[(guess_idx * 14 * MASK_WORDS) + (feedback * MASK_WORDS)];
        targetMask[word_idx] |= (1ULL << bit_idx);
    }
}