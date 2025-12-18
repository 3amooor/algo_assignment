// ArcadiaEngine.cpp

#include "ArcadiaEngine.h"
#include <algorithm>
#include <queue>
#include <numeric>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <set>

using namespace std;

// =========================================================
// HELPER FUNCTIONS
// =========================================================

// Helper function to convert long long to binary string
string toBinary(long long n) {
    if (n == 0) return "0";
    string binary = "";
    while (n > 0) {
        binary = (n % 2 == 0 ? "0" : "1") + binary;
        n /= 2;
    }
    return binary;
}

// =========================================================
// PART A: DATA STRUCTURES (Concrete Implementations)
// =========================================================

// --- 1. PlayerTable (Double Hashing) ---

class ConcretePlayerTable : public PlayerTable {
private:
    static const int TABLE_SIZE = 101;
    static const int PRIME = 97;

    struct Entry {
        int key;
        string value;
        bool occupied;
        bool deleted;

        Entry() : key(-1), value(""), occupied(false), deleted(false) {}
    };

    vector<Entry> table;

    int h1(int key) {
        return key % TABLE_SIZE;
    }

    int h2(int key) {
        return PRIME - (key % PRIME);
    }

public:
    ConcretePlayerTable() {
        table.resize(TABLE_SIZE);
    }

    void insert(int playerID, string name) override {
        int index = h1(playerID);
        int step = h2(playerID);

        for (int i = 0; i < TABLE_SIZE; i++) {
            int pos = (index + i * step) % TABLE_SIZE;

            if (!table[pos].occupied || table[pos].deleted) {
                table[pos].key = playerID;
                table[pos].value = name;
                table[pos].occupied = true;
                table[pos].deleted = false;
                return;
            }
        }
    }

    string search(int playerID) override {
        int index = h1(playerID);
        int step = h2(playerID);

        for (int i = 0; i < TABLE_SIZE; i++) {
            int pos = (index + i * step) % TABLE_SIZE;

            if (!table[pos].occupied && !table[pos].deleted)
                return "";

            if (table[pos].occupied && table[pos].key == playerID)
                return table[pos].value;
        }
        return "";
    }
};



// --- 2. Leaderboard (Skip List) ---

class ConcreteLeaderboard : public Leaderboard {
private:
    struct Node {
        int playerID;
        int score;
        vector<Node*> forward;

        Node(int level, int id, int s)
            : playerID(id), score(s), forward(level, nullptr) {}
    };

    static const int MAX_LEVEL = 6;
    const float P = 0.5f;

    int currentLevel;
    Node* header;

    int randomLevel() {
        int lvl = 1;
        while (((double)rand() / RAND_MAX) < P && lvl < MAX_LEVEL)
            lvl++;
        return lvl;
    }

public:
    ConcreteLeaderboard() {
        srand(time(nullptr));
        currentLevel = 1;
        header = new Node(MAX_LEVEL, -1, INT_MAX);
    }


    void addScore(int playerID, int score) override {
        vector<Node*> update(MAX_LEVEL);
        Node* cur = header;

        for (int i = currentLevel - 1; i >= 0; i--) {
            while (cur->forward[i] &&
                (cur->forward[i]->score > score ||
                    (cur->forward[i]->score == score &&
                        cur->forward[i]->playerID < playerID))) {
                cur = cur->forward[i];
            }
            update[i] = cur;
        }

        int lvl = randomLevel();
        if (lvl > currentLevel) {
            for (int i = currentLevel; i < lvl; i++)
                update[i] = header;
            currentLevel = lvl;
        }

        Node* node = new Node(lvl, playerID, score);
        for (int i = 0; i < lvl; i++) {
            node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = node;
        }
    }


    void removePlayer(int playerID) override {
        Node* target = nullptr;
        Node* cur = header;


        while (cur->forward[0]) {
            if (cur->forward[0]->playerID == playerID) {
                target = cur->forward[0];
                break;
            }
            cur = cur->forward[0];
        }

        if (!target) return;

        vector<Node*> update(MAX_LEVEL);
        cur = header;

        for (int i = currentLevel - 1; i >= 0; i--) {
            while (cur->forward[i] &&
                cur->forward[i] != target &&
                (cur->forward[i]->score > target->score ||
                    (cur->forward[i]->score == target->score &&
                        cur->forward[i]->playerID < target->playerID))) {
                cur = cur->forward[i];
            }
            update[i] = cur;
        }

        for (int i = 0; i < currentLevel; i++) {
            if (update[i]->forward[i] == target)
                update[i]->forward[i] = target->forward[i];
        }

        delete target;

        while (currentLevel > 1 &&
            header->forward[currentLevel - 1] == nullptr) {
            currentLevel--;
        }
    }


    vector<int> getTopN(int n) override {
        vector<int> result;
        Node* cur = header->forward[0];

        while (cur && n--) {
            result.push_back(cur->playerID);
            cur = cur->forward[0];
        }
        return result;
    }
};

// --- 3. AuctionTree (Red-Black Tree) ---

class ConcreteAuctionTree : public AuctionTree {
private:
    enum Color { RED, BLACK };

    struct Node {
        int id, price;
        Color color;
        Node* left;
        Node* right;
        Node* parent;

        Node(int i, int p)
            : id(i), price(p), color(RED),
            left(nullptr), right(nullptr), parent(nullptr) {}
    };

    Node* root;


    void rotateLeft(Node* x) {
        Node* y = x->right;
        x->right = y->left;
        if (y->left)
            y->left->parent = x;

        y->parent = x->parent;
        if (!x->parent)
            root = y;
        else if (x == x->parent->left)
            x->parent->left = y;
        else
            x->parent->right = y;

        y->left = x;
        x->parent = y;
    }

    void rotateRight(Node* y) {
        Node* x = y->left;
        y->left = x->right;
        if (x->right)
            x->right->parent = y;

        x->parent = y->parent;
        if (!y->parent)
            root = x;
        else if (y == y->parent->left)
            y->parent->left = x;
        else
            y->parent->right = x;

        x->right = y;
        y->parent = x;
    }

    void fixInsert(Node* z) {
        while (z->parent && z->parent->color == RED) {
            if (z->parent == z->parent->parent->left) {
                Node* y = z->parent->parent->right;
                if (y && y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                }
                else {
                    if (z == z->parent->right) {
                        z = z->parent;
                        rotateLeft(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    rotateRight(z->parent->parent);
                }
            }
            else {
                Node* y = z->parent->parent->left;
                if (y && y->color == RED) {
                    z->parent->color = BLACK;
                    y->color = BLACK;
                    z->parent->parent->color = RED;
                    z = z->parent->parent;
                }
                else {
                    if (z == z->parent->left) {
                        z = z->parent;
                        rotateRight(z);
                    }
                    z->parent->color = BLACK;
                    z->parent->parent->color = RED;
                    rotateLeft(z->parent->parent);
                }
            }
        }
        root->color = BLACK;
    }


    Node* searchByID(Node* x, int itemID) {
        if (!x) return nullptr;
        if (x->id == itemID) return x;

        Node* found = searchByID(x->left, itemID);
        if (found) return found;

        return searchByID(x->right, itemID);
    }

    Node* minimum(Node* x) {
        while (x->left)
            x = x->left;
        return x;
    }

    void transplant(Node* u, Node* v) {
        if (!u->parent)
            root = v;
        else if (u == u->parent->left)
            u->parent->left = v;
        else
            u->parent->right = v;

        if (v)
            v->parent = u->parent;
    }

    void fixDelete(Node* x) {
        while (x != root && (!x || x->color == BLACK)) {
            if (x == x->parent->left) {
                Node* w = x->parent->right;

                if (w && w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateLeft(x->parent);
                    w = x->parent->right;
                }

                if ((!w->left || w->left->color == BLACK) &&
                    (!w->right || w->right->color == BLACK)) {
                    w->color = RED;
                    x = x->parent;
                }
                else {
                    if (!w->right || w->right->color == BLACK) {
                        if (w->left) w->left->color = BLACK;
                        w->color = RED;
                        rotateRight(w);
                        w = x->parent->right;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    if (w->right) w->right->color = BLACK;
                    rotateLeft(x->parent);
                    x = root;
                }
            }
            else {
                Node* w = x->parent->left;

                if (w && w->color == RED) {
                    w->color = BLACK;
                    x->parent->color = RED;
                    rotateRight(x->parent);
                    w = x->parent->left;
                }

                if ((!w->right || w->right->color == BLACK) &&
                    (!w->left || w->left->color == BLACK)) {
                    w->color = RED;
                    x = x->parent;
                }
                else {
                    if (!w->left || w->left->color == BLACK) {
                        if (w->right) w->right->color = BLACK;
                        w->color = RED;
                        rotateLeft(w);
                        w = x->parent->left;
                    }
                    w->color = x->parent->color;
                    x->parent->color = BLACK;
                    if (w->left) w->left->color = BLACK;
                    rotateRight(x->parent);
                    x = root;
                }
            }
        }
        if (x) x->color = BLACK;
    }

public:
    ConcreteAuctionTree() : root(nullptr) {}

    void insertItem(int itemID, int price) override {
        Node* z = new Node(itemID, price);
        Node* y = nullptr;
        Node* x = root;

        while (x) {
            y = x;
            if (price < x->price ||
                (price == x->price && itemID < x->id))
                x = x->left;
            else
                x = x->right;
        }

        z->parent = y;
        if (!y)
            root = z;
        else if (price < y->price ||
            (price == y->price && itemID < y->id))
            y->left = z;
        else
            y->right = z;

        fixInsert(z);
    }


    void deleteItem(int itemID) override {
        Node* z = searchByID(root, itemID);
        if (!z) return;

        Node* y = z;
        Node* x = nullptr;
        Color yOriginalColor = y->color;

        if (!z->left) {
            x = z->right;
            transplant(z, z->right);
        }
        else if (!z->right) {
            x = z->left;
            transplant(z, z->left);
        }
        else {
            y = minimum(z->right);
            yOriginalColor = y->color;
            x = y->right;

            if (y->parent == z) {
                if (x) x->parent = y;
            }
            else {
                transplant(y, y->right);
                y->right = z->right;
                y->right->parent = y;
            }

            transplant(z, y);
            y->left = z->left;
            y->left->parent = y;
            y->color = z->color;
        }

        delete z;

        if (yOriginalColor == BLACK && x)
            fixDelete(x);
    }
};



// =========================================================
// PART B: INVENTORY SYSTEM (Dynamic Programming)
// =========================================================

int InventorySystem::optimizeLootSplit(int n, vector<int>& coins) {


    // Edge case: empty or single coin
    if (n == 0) return 0;
    if (n == 1) return coins[0];

    // Calculate total sum
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += coins[i];
    }

    // Target is to find closest sum to total/2
    int target = total / 2;

    // DP array: dp[i] = true if sum i is achievable
    vector<bool> dp(target + 1, false);
    dp[0] = true;

    // For each coin
    for (int i = 0; i < n; i++) {
        // Traverse from right to left to avoid using same coin twice
        for (int j = target; j >= coins[i]; j--) {
            if (dp[j - coins[i]]) {
                dp[j] = true;
            }
        }
    }

    // Find the largest sum <= target that is achievable
    int closestSum = 0;
    for (int i = target; i >= 0; i--) {
        if (dp[i]) {
            closestSum = i;
            break;
        }
    }

    int difference = abs(total - 2 * closestSum);

    return difference;
}

int InventorySystem::maximizeCarryValue(int capacity, vector<pair<int, int>>& items) {


    size_t n = items.size();

    // Edge cases
    if (n == 0 || capacity == 0) return 0;

    // DP array: dp[w] = maximum value achievable with capacity w
    // Using 1D array for space optimization
    vector<long long> dp(capacity + 1, 0);

    // For each item
    for (int i = 0; i < n; i++) {
        int weight = items[i].first;
        int value = items[i].second;

        // Traverse from right to left to avoid using same item twice
        for (int w = capacity; w >= weight; w--) {
            // Either take the item or don't take it
            dp[w] = max(dp[w], dp[w - weight] + value);
        }
    }

    return dp[capacity];
}

long long InventorySystem::countStringPossibilities(string s) {
    const long long mod = 1000000007;

    for (char c : s) {
        if (c == 'w' || c == 'm')
            return 0;
    }

    size_t n = s.size();
    vector<long long> dp(n + 1, 0);
    dp[0] = 1;

    for (int i = 1; i <= n; i++) {
        dp[i] = dp[i - 1] % mod;

        if (i >= 2) {
            string t = s.substr(i - 2, 2);
            if (t == "uu" || t == "nn") {
                dp[i] = (dp[i] + dp[i - 2]) % mod;
            }
        }
    }
    return dp[n];
}

// =========================================================
// PART C: WORLD NAVIGATOR (Graphs)
// =========================================================

bool WorldNavigator::pathExists(int n, vector<vector<int>>& edges, int source, int dest) {
    if (source == dest) return true;

    vector<vector<int>> adj(n);
    for (auto& e : edges) {
        adj[e[0]].push_back(e[1]);
        adj[e[1]].push_back(e[0]); 
    }

    // All nodes start as WHITE 
    vector<bool> visited(n, false);
    queue<int> q;

    // Source becomes GRAY: discovered and added to queue
    q.push(source);
    visited[source] = true;

    while (!q.empty()) {

        // Take a GRAY node from the queue
        int u = q.front();
        q.pop();

        for (int v : adj[u]) {
            if (!visited[v]) {
                if (v == dest) return true;

                // Neighbor becomes GRAY
                visited[v] = true;
                q.push(v);
            }
        }
    }

    return false;
}


long long WorldNavigator::minBribeCost(int n, int m,
                                       long long goldRate, long long silverRate,
                                       vector<vector<int>>& roadData) {

    vector<int> parent(n), rank(n, 0);
    for (int i = 0; i < n; i++)
        parent[i] = i;

    // Find with path compression
    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;

        if (rank[a] < rank[b]) swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) rank[a]++;
        return true;
    };

    vector<tuple<long long, int, int>> edges;
    for (auto& r : roadData) {
        long long cost = r[2] * goldRate + r[3] * silverRate;
        edges.push_back({cost, r[0], r[1]});
    }

    sort(edges.begin(), edges.end());

    long long totalCost = 0;
    int usedEdges = 0;

    for (auto& [cost, u, v] : edges) {
        if (unite(u, v)) {
            totalCost += cost;
            usedEdges++;
        }
    }

    return (usedEdges == n - 1) ? totalCost : -1;
}


string WorldNavigator::sumMinDistancesBinary(int n, vector<vector<int>>& roads) {
    // All-Pairs Shortest Path using Floyd-Warshall Algorithm
    const long long INF = 1e18;
    vector<vector<long long>> dist(n, vector<long long>(n, INF));

    // 1. Initialize self-distance to 0
    for (int i = 0; i < n; ++i) {
        dist[i][i] = 0;
    }

    // 2. Populate initial distances from road data
    for (const auto& road : roads) {
        int u = road[0];
        int v = road[1];
        int w = road[2];
        // Use the smallest weight if multiple roads exist
        if (w < dist[u][v]) {
            dist[u][v] = w;
            dist[v][u] = w; // Undirected graph
        }
    }

    // 3. Floyd-Warshall Algorithm
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] != INF && dist[k][j] != INF) {
                    if (dist[i][k] + dist[k][j] < dist[i][j]) {
                        dist[i][j] = dist[i][k] + dist[k][j];
                    }
                }
            }
        }
    }

    // 4. Sum all shortest distances between unique pairs (i < j)
    long long totalSum = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (dist[i][j] != INF) {
                totalSum += dist[i][j];
            }
        }
    }

    // 5. Convert sum to binary string
    return toBinary(totalSum);
}

// =========================================================
// PART D: SERVER KERNEL (Greedy)
// =========================================================

int ServerKernel::minIntervals(vector<char>& tasks, int n) {
    // Greedy Task Scheduler Strategy

    // 1. Count frequency of each task
    map<char, int> counts;
    for (char t : tasks) {
        counts[t]++;
    }

    // 2. Find the maximum frequency among all tasks
    int maxFreq = 0;
    for (auto const& [key, val] : counts) {
        if (val > maxFreq) {
            maxFreq = val;
        }
    }

    // 3. Count how many tasks share this maximum frequency
    int numTasksWithMaxFreq = 0;
    for (auto const& [key, val] : counts) {
        if (val == maxFreq) {
            numTasksWithMaxFreq++;
        }
    }

    // 4. Calculate minimum intervals
    // Formula: (maxFreq - 1) groups of size (n + 1) + remaining tasks with max frequency
    long long minLen = (long long)(maxFreq - 1) * (n + 1) + numTasksWithMaxFreq;

    // 5. Result cannot be less than the total number of tasks
    return max((int)minLen, (int)tasks.size());
}

// =========================================================
// FACTORY FUNCTIONS (Required for Testing)
// =========================================================

extern "C" {
PlayerTable* createPlayerTable() {
    return new ConcretePlayerTable();
}

Leaderboard* createLeaderboard() {
    return new ConcreteLeaderboard();
}

AuctionTree* createAuctionTree() {
    return new ConcreteAuctionTree();
}
}