/**
 * main_test_student.cpp
 * Basic "Happy Path" Test Suite for ArcadiaEngine
 * Use this to verify your basic logic against the assignment examples.
 */

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <functional>
#include "ArcadiaEngine.h" 

using namespace std;

// ==========================================
// FACTORY FUNCTIONS (LINKING)
// ==========================================
// These link to the functions at the bottom of your .cpp file
extern "C" {
    PlayerTable* createPlayerTable();
    Leaderboard* createLeaderboard();
    AuctionTree* createAuctionTree();
}

// ==========================================
// TEST UTILITIES
// ==========================================
class StudentTestRunner {
	int count = 0;
    int passed = 0;
    int failed = 0;

public:
    void runTest(string testName, bool condition) {
		count++;
        cout << "TEST: " << left << setw(50) << testName;
        if (condition) {
            cout << "[ PASS ]";
            passed++;
        } else {
            cout << "[ FAIL ]";
            failed++;
        }
        cout << endl;
    }

    void printSummary() {
        cout << "\n==========================================" << endl;
        cout << "SUMMARY: Passed: " << passed << " | Failed: " << failed << endl;
        cout << "==========================================" << endl;
		cout << "TOTAL TESTS: " << count << endl;
        if (failed == 0) {
            cout << "Great job! All basic scenarios passed." << endl;
            cout << "Now make sure to handle edge cases (empty inputs, collisions, etc.)!" << endl;
        } else {
            cout << "Some basic tests failed. Check your logic against the PDF examples." << endl;
        }
    }
};

StudentTestRunner runner;

// ==========================================
// PART A: DATA STRUCTURES HARD TESTS
// ==========================================
void test_Hard_PartA() {
    cout << "\n--- Part A: Hard Tests ---" << endl;

    // --- PlayerTable ---
    PlayerTable* table = createPlayerTable();
    table->insert(101, "Alice");
    table->insert(202, "Bob");   // Collision
    table->insert(303, "Charlie"); // Collision
    runner.runTest("PlayerTable: Collisions Insert/Search",
        table->search(101)=="Alice" &&
        table->search(202)=="Bob" &&
        table->search(303)=="Charlie");
    runner.runTest("PlayerTable: Search missing key", table->search(404) == "");
    delete table;

    // --- Leaderboard ---
    Leaderboard* board = createLeaderboard();
    board->addScore(1, 500);
    board->addScore(2, 500);
    board->addScore(3, 500);
    board->addScore(4, 400);
    runner.runTest("Leaderboard: Tie-breaking top 3",
        [&]() { vector<int> top=board->getTopN(3); return top[0]==1 && top[1]==2 && top[2]==3; }());
    board->removePlayer(2);
    runner.runTest("Leaderboard: Remove player ID 2",
        [&]() { vector<int> top=board->getTopN(3); return top[0]==1 && top[1]==3; }());
    delete board;

    // --- AuctionTree ---
    AuctionTree* tree = createAuctionTree();
    tree->insertItem(1, 100);
    tree->insertItem(2, 100);
    tree->insertItem(3, 50);
    tree->insertItem(4, 150);
    tree->deleteItem(1);
    runner.runTest("AuctionTree: Delete root node", true);
    tree->deleteItem(3);
    runner.runTest("AuctionTree: Delete leaf node", true);
    delete tree;
}

// ==========================================
// PART B: INVENTORY SYSTEM HARD TESTS
// ==========================================
void test_Hard_PartB() {
    cout << "\n--- Part B: Hard Tests ---" << endl;

    // LootSplit
    vector<int> coins = {100,200,300,400,500};
    runner.runTest("LootSplit: Large values",
        InventorySystem::optimizeLootSplit(coins.size(), coins) == 100);

    coins = {1,1,1,1,1,1,1,1};
    runner.runTest("LootSplit: Many duplicates",
        InventorySystem::optimizeLootSplit(coins.size(), coins) == 0);

    // Knapsack
    vector<pair<int,int>> items;
    for(int i=1;i<=20;i++) items.push_back({i,i*10});
    runner.runTest("Knapsack: Many items, large capacity",
        InventorySystem::maximizeCarryValue(50, items) == 500);

    // Chat Decoder
    runner.runTest("ChatDecoder: Complex string 'uunnuu'",
        InventorySystem::countStringPossibilities("uunnuu") == 8);

    runner.runTest("ChatDecoder: String with 'w'",
        InventorySystem::countStringPossibilities("w") == 0);
}

// ==========================================
// PART C: WORLD NAVIGATOR HARD TESTS
// ==========================================
void test_Hard_PartC() {
    cout << "\n--- Part C: Hard Tests ---" << endl;

    // PathExists
    vector<vector<int>> edges = {{0,1},{2,3},{3,4}};
    runner.runTest("PathExists: Disconnected components",
        WorldNavigator::pathExists(5, edges, 0, 4) == false);

    edges.clear();
    for(int i=0;i<99;i++) edges.push_back({i,i+1});
    runner.runTest("PathExists: Linear 100 nodes",
        WorldNavigator::pathExists(100, edges, 0, 99) == true);

    // MinBribeCost
    edges = {{0,1,1,1},{2,3,2,2}};
    runner.runTest("MinBribeCost: Disconnected graph",
        WorldNavigator::minBribeCost(4,2,1,1,edges) == -1);

    edges = {{0,1,1,1},{1,2,2,2},{0,2,3,3},{2,3,4,1}};
    runner.runTest("MinBribeCost: Small connected graph",
        WorldNavigator::minBribeCost(4,4,1,1,edges) == 11);

    // SumMinDistancesBinary
    edges = {{0,1,1},{1,2,2},{2,3,3},{0,3,10}};
    runner.runTest("BinarySum: Medium graph",
        WorldNavigator::sumMinDistancesBinary(4, edges) == "10100"); // sum=15

    edges.clear();
    int n = 10;
    for(int i=0;i<n-1;i++) edges.push_back({i,i+1,1});
    runner.runTest("BinarySum: Linear 10 nodes",
        WorldNavigator::sumMinDistancesBinary(10, edges) == "10100101"); // sum=45 -> "101101"
}

// ==========================================
// PART D: SERVER KERNEL HARD TESTS
// ==========================================
void test_Hard_PartD() {
    cout << "\n--- Part D: Hard Tests ---" << endl;

    runner.runTest("Scheduler: High Freq + Cooldown",
        [&]() {
            vector<char> tasks={'A','A','A','B','B','C'};
            int n=3;
            return ServerKernel::minIntervals(tasks,n)==9;
        }());

    runner.runTest("Scheduler: Single Task Repeats",
        [&]() {
            vector<char> tasks={'A','A','A','A'};
            int n=5;
            return ServerKernel::minIntervals(tasks,n)==19;
        }());

    runner.runTest("Scheduler: Multiple Max Frequency Tasks",
        [&]() {
            vector<char> tasks={'A','A','B','B','C','C'};
            int n=2;
            return ServerKernel::minIntervals(tasks,n)==6;
        }());

    runner.runTest("Scheduler: n=0 No Cooldown",
        [&]() {
            vector<char> tasks={'A','A','A','B','B','C'};
            int n=0;
            return ServerKernel::minIntervals(tasks,n)==6;
        }());
}

// ==========================================
// MAIN
// ==========================================
int main() {
    cout << "Arcadia Engine - Hard Tests" << endl;
    cout << "-----------------------------------------" << endl;

    test_Hard_PartA();
    test_Hard_PartB();
    test_Hard_PartC();
    test_Hard_PartD();

    runner.printSummary();
    return 0;
}