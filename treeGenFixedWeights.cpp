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
const int MAX_GAME_LENGTH = 12;
const int PUNISHMENT_SCORE = 13;

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
std::pair<int, int> getFeedbackBullsCows(int guess_code, int secret_code);
double calculateWeightedShannon(const std::vector<int>& counts, int total, const double* weights);
int findBestGuess(const GameState& state, const std::array<int, S>& valids, const double* weights);
int playGame(int secret_idx, const std::array<int, S>& valids, const double* weights);

// --- GLOBAL STATISTICS ---
std::array<int, MAX_GAME_LENGTH + 1> game_stats = { 0 };

// --- MAIN EXECUTION ---
int main() {
    // --- FIXED WEIGHTS FOR ALL TURNS ---
    const double fixed_weights[NUM_WEIGHTS] = {
        0.473, 0.446, 0.523, 0.41, 0.35, 0.534, 0.486, 0.423, 0.383, 0.406, 0.413, 0.458, 0.424, 0.8
    };

    auto valids = generateValids();
    long long total_turns = 0;

    std::cout << "--- Mastermind Solver Validation (Fixed Weighted Shannon Entropy) ---" << std::endl;
    std::cout << "Playing all " << S << " games with fixed weights across all turns..." << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(8) << "Secret" << " | " << "Guess Sequence (Bulls/Cows) -> Turns" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int i = 0; i < S; ++i) {
        total_turns += playGame(i, valids, fixed_weights);
    }

    double average_turns = static_cast<double>(total_turns) / S;

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "All games completed." << std::endl;
    std::cout << "Total turns for " << S << " games: " << total_turns << std::endl;
    std::cout << "Final Average Game Length: " << std::fixed << std::setprecision(8) << average_turns << " turns." << std::endl;

    // Print detailed statistics
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Game Length Statistics:" << std::endl;
    for (int i = 1; i <= MAX_GAME_LENGTH; ++i) {
        double percentage = (static_cast<double>(game_stats[i]) / S) * 100.0;
        std::cout << "  " << i << " turn" << (i > 1 ? "s" : " ") << ": "
            << std::setw(4) << game_stats[i] << " games ("
            << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
    }
    if (game_stats[0] > 0) { // failures stored at index 0 for convenience
        double percentage = (static_cast<double>(game_stats[0]) / S) * 100.0;
        std::cout << "  Failed: " << std::setw(4) << game_stats[0] << " games ("
            << std::fixed << std::setprecision(2) << percentage << "%)" << std::endl;
    }

    return 0;
}

// --- CORE LOGIC FUNCTIONS ---

int playGame(int secret_idx, const std::array<int, S>& valids, const double* weights) {
    GameState state;
    int turn = 1;

    // Use weights to find the best first guess instead of forcing 1123
    int current_guess_idx = findBestGuess(state, valids, weights);

    std::cout << std::setw(8) << valids[secret_idx] << " | ";
    std::cout << std::flush;

    while (turn <= MAX_GAME_LENGTH) {
        std::cout << valids[current_guess_idx];

        // Add bulls/cows feedback if not the winning guess
        if (current_guess_idx != secret_idx) {
            auto [bulls, cows] = getFeedbackBullsCows(valids[current_guess_idx], valids[secret_idx]);
            std::cout << "(" << bulls << "/" << cows << ")";
        }

        std::cout << " ";
        std::cout << std::flush;
        state.previous_guesses.push_back(current_guess_idx);

        if (current_guess_idx == secret_idx) {
            std::cout << "-> " << turn << std::endl;
            game_stats[turn]++; // Track successful completion
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

        current_guess_idx = findBestGuess(state, valids, weights);
    }

    std::cout << "-> " << PUNISHMENT_SCORE << " (FAIL)" << std::endl;
    game_stats[0]++; // Track failures at index 0
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
        scores[guess_idx] = calculateWeightedShannon(partition_counts, state.num_possible, weights);
    }

    double max_score = -1e8;
    for (double score : scores) if (score > max_score) max_score = score;

    for (int i = 0; i < S; ++i) if (std::abs(scores[i] - max_score) < 1e-6 && state.possible_codes[i]) return i;
    for (int i = 0; i < S; ++i) if (std::abs(scores[i] - max_score) < 1e-6) return i;
    return -1;
}

double calculateWeightedShannon(const std::vector<int>& counts, int total, const double* weights) {
    double shannon = 0.0;
    if (total == 0) return 0.0;

    for (int i = 0; i < NUM_WEIGHTS; ++i) {
        if (counts[i] > 0) {
            double p = static_cast<double>(counts[i]) / total;
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

std::pair<int, int> getFeedbackBullsCows(int guess_code, int secret_code) {
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

    return { p, m }; // Return bulls and cows as a pair
}