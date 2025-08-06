#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>

// --- CORE GAME CONSTANTS ---
const int S = 1296;
const int NUM_WEIGHTS = 14;
const int FORCED_FIRST_GUESS_IDX = 8; // 1123
const int MAX_GAME_LENGTH = 6;
const int PUNISHMENT_SCORE = 8;

// --- DATA STRUCTURES ---
struct GameState {
    std::vector<bool> possible_codes;
    std::vector<int> previous_guesses;
    int num_possible;

    GameState() : possible_codes(S, true), num_possible(S) {}
};

// --- FORWARD DECLARATIONS ---
std::array<int, S> generateValids();
int calculateFeedbackIndex(int guess_code, int secret_code);
double calculateWeightedShannon(const std::vector<int>& counts, int total, const double* weights);
int findBestGuess(const GameState& state, const std::array<int, S>& valids, const double* weights);
int playGame(int secret_idx, const std::array<int, S>& valids, const std::vector<const double*>& all_weights);

// --- MAIN EXECUTION ---
int main() {
    // --- USER-DEFINED WEIGHTS ---
    // Paste the optimal weights discovered by the CUDA optimizer here.
    // This set corresponds to the 4.3488 average.
    /*const double weights_turn2[NUM_WEIGHTS] = { 0.6459,0.5739,0.5865,0.5061,0.4330,0.6108,0.8458,0.6003,0.3230,0.3440,0.4271,0.6429,0.3841,0.9822 };
    const double weights_turn3[NUM_WEIGHTS] = { 0.6454,0.4107,0.5261,0.4655,0.3692,0.4205,0.4697,0.5121,0.4582,0.4764,0.4644,0.4632,0.4950,0.8572 };
    const double weights_turn4[NUM_WEIGHTS] = { 0.2540,0.4965,0.3954,0.4763,0.4147,0.4585,0.4805,0.3894,0.6161,0.3748,0.4473,0.4636,0.5051,0.9904 };
    const double weights_turn5[NUM_WEIGHTS] = { 0.3694,0.4989,0.3168,0.6088,0.4599,0.4073,0.5367,0.5334,0.4450,0.5474,0.5710,0.6476,0.5788,0.7618 };
    const double weights_turn6[NUM_WEIGHTS] = { 0.2314,0.7670,0.4116,0.5980,0.6434,0.6072,0.7050,0.4588,0.1644,0.6098,0.4054,0.3377,0.4642,0.4427 };*/

    // This much rounded set still gives 4.3488 average!
   /* const double weights_turn2[NUM_WEIGHTS] = { 0.7, 0.6, 0.6, 0.51, 0.43, 0.6, 0.85, 0.6, 0.32, 0.34, 0.4, 0.6, 0.4, 1.0 };
    const double weights_turn3[NUM_WEIGHTS] = { 0.7, 0.41, 0.53, 0.47, 0.37, 0.4, 0.47, 0.5, 0.46, 0.48, 0.46, 0.5, 0.5, 0.9 };
    const double weights_turn4[NUM_WEIGHTS] = { 0.3, 0.5, 0.4, 0.5, 0.4, 0.5, 0.5, 0.4, 0.6, 0.4, 0.5, 0.5, 0.5, 1.0 };
    const double weights_turn5[NUM_WEIGHTS] = { 0.4, 0.6, 0.3, 0.6, 0.5, 0.4, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 0.6, 0.8 };
    const double weights_turn6[NUM_WEIGHTS] = { 0.2, 0.8, 0.4, 0.6, 0.6, 0.6, 0.7, 0.5, 0.2, 0.6, 0.4, 0.3, 0.5, 0.4 };*/

    //old 4.3565 weights
    const double weights_turn2[NUM_WEIGHTS] = { 0.4733,0.4458,0.5233,0.4082,0.3486,0.5340,0.4862,0.4233,0.3833,0.4064,0.4133,0.4568,0.4238,0.7977 };
    const double weights_turn3[NUM_WEIGHTS] = { 0.4733,0.4458,0.5233,0.4082,0.3486,0.5340,0.4862,0.4233,0.3833,0.4064,0.4133,0.4568,0.4238,0.7977 };
    const double weights_turn4[NUM_WEIGHTS] = { 0.4733,0.4458,0.5233,0.4082,0.3486,0.5340,0.4862,0.4233,0.3833,0.4064,0.4133,0.4568,0.4238,0.7977 };
    const double weights_turn5[NUM_WEIGHTS] = { 0.4733,0.4458,0.5233,0.4082,0.3486,0.5340,0.4862,0.4233,0.3833,0.4064,0.4133,0.4568,0.4238,0.7977 };
    const double weights_turn6[NUM_WEIGHTS] = { 0.4733,0.4458,0.5233,0.4082,0.3486,0.5340,0.4862,0.4233,0.3833,0.4064,0.4133,0.4568,0.4238,0.7977 };



    // A vector of pointers to easily access weights by turn number
    std::vector<const double*> all_weights = {
        nullptr, // Turn 0 (unused)
        nullptr, // Turn 1 (fixed guess)
        weights_turn2,
        weights_turn3,
        weights_turn4,
        weights_turn5,
        weights_turn6
    };

    auto valids = generateValids();
    long long total_turns = 0;

    std::cout << "--- Mastermind Solver Validation ---" << std::endl;
    std::cout << "Playing all " << S << " games with the provided optimal weights..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(8) << "Secret" << " | " << "Guess Sequence -> Turns" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int i = 0; i < S; ++i) {
        total_turns += playGame(i, valids, all_weights);
    }

    double average_turns = static_cast<double>(total_turns) / S;

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "All games completed." << std::endl;
    std::cout << "Total turns for " << S << " games: " << total_turns << std::endl;
    std::cout << "Final Average Game Length: " << std::fixed << std::setprecision(8) << average_turns << " turns." << std::endl;

    return 0;
}

// --- CORE LOGIC FUNCTIONS (Ported from CUDA) ---

int playGame(int secret_idx, const std::array<int, S>& valids, const std::vector<const double*>& all_weights) {
    GameState state;
    int turn = 1;
    int current_guess_idx = FORCED_FIRST_GUESS_IDX;

    std::cout << std::setw(8) << valids[secret_idx] << " | ";
    std::cout << std::flush;

    while (turn <= MAX_GAME_LENGTH) {
        std::cout << valids[current_guess_idx] << " ";
        std::cout << std::flush;
        state.previous_guesses.push_back(current_guess_idx);

        if (current_guess_idx == secret_idx) {
            std::cout << "-> " << turn << std::endl;
            return turn;
        }

        int feedback = calculateFeedbackIndex(valids[current_guess_idx], valids[secret_idx]);

        int possible_count = 0;
        for (int i = 0; i < S; ++i) {
            if (state.possible_codes[i]) {
                if (calculateFeedbackIndex(valids[current_guess_idx], valids[i]) != feedback) {
                    state.possible_codes[i] = false;
                }
                else {
                    possible_count++;
                }
            }
        }
        state.num_possible = possible_count;

        turn++;
        if (turn > MAX_GAME_LENGTH) break;

        current_guess_idx = findBestGuess(state, valids, all_weights[turn]);
    }

    std::cout << "-> " << PUNISHMENT_SCORE << " (FAIL)" << std::endl;
    return PUNISHMENT_SCORE;
}

int findBestGuess(const GameState& state, const std::array<int, S>& valids, const double* weights) {
    if (state.num_possible <= 1) {
        for (int i = 0; i < S; ++i) {
            if (state.possible_codes[i]) {
                bool is_previous = false;
                for (int prev : state.previous_guesses) if (i == prev) is_previous = true;
                if (!is_previous) return i;
            }
        }
    }

    std::vector<double> scores(S);
    for (int guess_idx = 0; guess_idx < S; ++guess_idx) {
        bool is_previous = false;
        for (int prev : state.previous_guesses) if (guess_idx == prev) is_previous = true;
        if (is_previous) { scores[guess_idx] = -1e9; continue; }

        std::vector<int> partition_counts(NUM_WEIGHTS, 0);
        for (int secret_idx = 0; secret_idx < S; ++secret_idx) {
            if (state.possible_codes[secret_idx]) {
                int feedback = calculateFeedbackIndex(valids[guess_idx], valids[secret_idx]);
                partition_counts[feedback]++;
            }
        }
        // FIX: Pass the weights to the calculation function
        scores[guess_idx] = calculateWeightedShannon(partition_counts, state.num_possible, weights);
    }

    double max_score = -1e8;
    for (double score : scores) if (score > max_score) max_score = score;

    for (int i = 0; i < S; ++i) if (std::abs(scores[i] - max_score) < 1e-6 && state.possible_codes[i]) return i;
    for (int i = 0; i < S; ++i) if (std::abs(scores[i] - max_score) < 1e-6) return i;
    return -1;
}

// FIX: This function now correctly implements WEIGHTED Shannon entropy
double calculateWeightedShannon(const std::vector<int>& counts, int total, const double* weights) {
    double shannon = 0.0;
    if (total == 0) return 0.0;

    for (int i = 0; i < NUM_WEIGHTS; ++i) {
        if (counts[i] > 0) {
            double p = static_cast<double>(counts[i]) / total;
            // The crucial fix: multiply by the corresponding weight
            shannon -= weights[i] * p * std::log2(p);
        }
    }
    return shannon;
}

std::array<int, S> generateValids() {
    std::array<int, S> valids_array{};
    int index = 0;
    for (int d1 = 1; d1 <= 6; ++d1)
        for (int d2 = 1; d2 <= 6; ++d2)
            for (int d3 = 1; d3 <= 6; ++d3)
                for (int d4 = 1; d4 <= 6; ++d4)
                    valids_array[index++] = d1 * 1000 + d2 * 100 + d3 * 10 + d4;
    return valids_array;
}

int calculateFeedbackIndex(int guess_code, int secret_code) {
    if (guess_code == secret_code) return 13;
    int p = 0, m = 0;
    std::array<int, 4> guess_digits, secret_digits;

    // Unpack digits
    int temp_guess = guess_code;
    int temp_secret = secret_code;
    for (int i = 3; i >= 0; --i) {
        guess_digits[i] = temp_guess % 10;
        secret_digits[i] = temp_secret % 10;
        temp_guess /= 10;
        temp_secret /= 10;
    }

    // First pass for perfect matches (bulls)
    for (int i = 0; i < 4; ++i) {
        if (guess_digits[i] == secret_digits[i]) {
            p++;
            guess_digits[i] = 0; // Mark as used
            secret_digits[i] = -1; // Mark as used
        }
    }

    // Second pass for color matches (cows)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (guess_digits[i] != 0 && guess_digits[i] == secret_digits[j]) {
                m++;
                secret_digits[j] = -1; // Mark as used
                break;
            }
        }
    }
    return static_cast<int>(-0.5f * p * p + 5.5f * p + m);
}