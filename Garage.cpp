#include<bits/stdc++.h>
#define ll long long
// #define mod 1000000007
using namespace std;
const int N = 3e5;
int dx[4]={1,0,-1,0},dy[4]={0,1,0,-1};  
int ddx[8]={1,1,0,-1,-1,-1,0,1},ddy[8]={0,1,1,1,0,-1,-1,-1};
int gcd(int a,int b){ if(!a)return b;return gcd(b%a,a);}
int lcm(int a, int b) { return (a*b)/ gcd(a,b);}


string toBinary(int n)
{
    string r;
    while(n!=0) {r=(n%2==0 ?"0":"1")+r; n/=2;}
    return r;
}


bool isVowel(char x) {
    return x == 'a' || x == 'e' || x == 'i' || x == 'o' || x == 'u';
}


const int mod = 1e9 + 7;

// a power to b mod m
// (a^b) % m

int binpow(int a, int b) {
    a %= mod;
    int res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res;
}
 
int mul(int a, int b) {
    return ( a % mod ) * (b %  mod) % mod;
}
 
int add(int a, int b) {
    return (a%mod+ b%mod) % mod;
}
 
int sub(int a, int b) {
    return (a%mod - b%mod + mod) % mod;
}
 
int divide(int a, int b) {
    return  a * (int)binpow(b,mod-2) % mod;
}



struct Sieve {
  int n;
  vector<int> f, primes;
  Sieve(int n=1):n(n), f(n+1) {
    f[0] = f[1] = -1;
    for (ll i = 2; i <= n; ++i) {
      if (f[i]) continue;
      primes.push_back(i);
      f[i] = i;
      for (ll j = i*i; j <= n; j += i) {
        if (!f[j]) f[j] = i;
      }
    }
  }
  bool isPrime(int x) { return f[x] == x;}
  vector<int> factorList(int x) {
    vector<int> res;
    while (x != 1) {
      res.push_back(f[x]);
      x /= f[x];
    }
    return res;
  }

  vector<pair<int,int> > factor(int x) {
    vector<int> fl = factorList(x);
    if (fl.size() == 0) return vector<pair<int,int> >();
    vector<pair<int,int> > res(1, make_pair(fl[0], 0));
    for (int p : fl) {
      if (res.back().first == p) {
        res.back().second++;
      } else {
        res.push_back(make_pair(p, 1));
      }
    }
    return res;
  }
};




bool isprime(ll n) //Time Complexity--->sqrt(n)
{   if (n <= 1)  return false; if (n <= 3)  return true; 
  	if (n%2 == 0 || n%3 == 0) return false; 
    for (int i=5; i*i<=n; i=i+6) if (n%i == 0 || n%(i+2) == 0) return false; 
    return true; 
} 


bool ispower2(int x) {
    return x && (!(x&(x-1)));
}



// GCD

int gcd(int a, int b) {
    if (!a || !b)
        return a | b;
    unsigned shift = __builtin_ctz(a | b);
    a >>= __builtin_ctz(a);
    do {
        b >>= __builtin_ctz(b);
        if (a > b)
            swap(a, b);
        b -= a;
    } while (b);
    return a << shift;
}

// Fcatorial

ll fact(ll n) 
{ 
    ll res = 1; 
    for (ll i = 2; i <= n; i++) 
        res = res * i; 
    return res; 
} 


// Calcculate fibonnaci

pair<int, int> fib (int n) {
    if (n == 0)
        return {0, 1};

    auto p = fib(n >> 1);
    int c = p.first * (2 * p.second - p.first);
    int d = p.first * p.first + p.second * p.second;
    if (n & 1)
        return {d, c + d};
    else
        return {c, d};
}

/*

using this formula

F(2k) = F(k) * ( 2*F(k+1) - F(k))

F(2k+1) = F(k+1) * F(k+1) + F(k) * F(k)


*/


// Gcd with coeff

int gcd(int a, int b, int& x, int& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    int x1, y1;
    int d = gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return d;
}

// Iterative

int gcd(int a, int b, int& x, int& y) {
    x = 1, y = 0;
    int x1 = 0, y1 = 1, a1 = a, b1 = b;
    while (b1) {
        int q = a1 / b1;
        tie(x, x1) = make_tuple(x1, x - q * x1);
        tie(y, y1) = make_tuple(y1, y - q * y1);
        tie(a1, b1) = make_tuple(b1, a1 - q * b1);
    }
    return a1;
}


// Find any solution of ax + by = c

int gcd(int a, int b, int& x, int& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    int x1, y1;
    int d = gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return d;
}




// check whether the number is prime or not 



// using u64 = uint64_t;
// using u128 = __uint128_t;

// u64 binpower(u64 base, u64 e, u64 mod) {
//     u64 result = 1;
//     base %= mod;
//     while (e) {
//         if (e & 1)
//             result = (u128)result * base % mod;
//         base = (u128)base * base % mod;
//         e >>= 1;
//     }
//     return result;
// }






// bool check_composite(u64 n, u64 a, u64 d, int s) {
//     u64 x = binpower(a, d, n);
//     if (x == 1 || x == n - 1)
//         return false;
//     for (int r = 1; r < s; r++) {
//         x = (u128)x * x % n;
//         if (x == n - 1)
//             return false;
//     }
//     return true;
// };


// bool MillerRabin(u64 n) { // returns true if n is prime, else returns false.
//     if (n < 2)
//         return false;

//     int r = 0;
//     u64 d = n - 1;
//     while ((d & 1) == 0) {
//         d >>= 1;
//         r++;
//     }

//     for (int a : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}) {
//         if (n == a)
//             return true;
//         if (check_composite(n, a, d, r))
//             return false;
//     }
//     return true;
// }



// Sieve of Eratosthenes
// https://youtu.be/UTVg7wzMWQc?t=2774
struct Sieve {
  int n;
  vector<int> f, primes;
  Sieve(int n=1):n(n), f(n+1) {
    f[0] = f[1] = -1;
    for (ll i = 2; i <= n; ++i) {
      if (f[i]) continue;
      primes.push_back(i);
      f[i] = i;
      for (ll j = i*i; j <= n; j += i) {
        if (!f[j]) f[j] = i;
      }
    }
  }
  bool isPrime(int x) { return f[x] == x;}
  vector<int> factorList(int x) {
    vector<int> res;
    while (x != 1) {
      res.push_back(f[x]);
      x /= f[x];
    }
    return res;
  }

  vector<pair<int,int> > factor(int x) {
    vector<int> fl = factorList(x);
    if (fl.size() == 0) return vector<pair<int,int> >();
    vector<pair<int,int> > res(1, make_pair(fl[0], 0));
    for (int p : fl) {
      if (res.back().first == p) {
        res.back().second++;
      } else {
        res.push_back(make_pair(p, 1));
      }
    }
    return res;
  }
};

// ncr of large numbers


const int MAX = 510000;
long long fac[MAX], finv[MAX], inv[MAX];
 
void COMinit() {
  fac[0] = fac[1] = 1;
  finv[0] = finv[1] = 1;
  inv[1] = 1;
  for (int i = 2; i < MAX; i++){
    fac[i] = fac[i - 1] * i % mod;
    inv[i] = mod - inv[mod%i] * (mod / i) % mod;
    finv[i] = finv[i - 1] * inv[i] % mod;
  }
}
 
 
 
long long COM(int n, int k){
  if (n < k) return 0;
  if (n < 0 || k < 0) return 0;
  return fac[n] * (finv[k] * finv[n - k] % mod) % mod;
}



// Factors count 
int trial_division4(long long n) {
    vector<int> primes; // Get it using sieves
    int cnt = 0;
    int ans = 0;
    for (long long d : primes) {
        if (d * d > n)
            break;
        while (n % d == 0) {
            cnt++;
            n /= d;
        }
    }
    if (n > 1)
        cnt++;
    
    return cnt;
}

// Number of factors


const int MAX = 1000001; 
int factor[MAX] = { 0 }; 
  

void generatePrimeFactors() 
{ 
    factor[1] = 1; 
  
    // Initializes all the positions with their value. 
    for (int i = 2; i < MAX; i++) 
        factor[i] = i; 
  
    // Initializes all multiples of 2 with 2 
    for (int i = 4; i < MAX; i += 2) 
        factor[i] = 2; 
  
    // A modified version of Sieve of Eratosthenes to 
    // store the smallest prime factor that divides 
    // every number. 
    for (int i = 3; i * i < MAX; i++) { 
        // check if it has no prime factor. 
        if (factor[i] == i) { 
            // Initializes of j starting from i*i 
            for (int j = i * i; j < MAX; j += i) { 
                // if it has no prime factor before, then 
                // stores the smallest prime divisor 
                if (factor[j] == j) 
                    factor[j] = i; 
            } 
        } 
    } 
} 
  
// function to calculate number of factors 


int calculateNoOFactors(int n) 
{ 
    if (n == 1) 
        return 1; 
  
    int ans = 1; 
  

    int dup = factor[n]; 

    int c = 1; 
  

    int j = n / factor[n]; 

    while (j != 1) { 

        if (factor[j] == dup) 
            c += 1; 

        else { 
            dup = factor[j]; 
            ans = ans * (c + 1); 
            c = 1; 
        } 
  
        j = j / factor[j]; 
    } 
   
    ans = ans * (c + 1); 
  
    return ans; 
} 
 


// LIS in Nlogn

int LIS(vector<int> array, int n)  {
    vector<int > ans;
    for (int i = 0; i < n; i++) {
        int x = array[i];
        //change lower_bound to upper_bound if strictly increasing is not important
        vector<int>::iterator it = lower_bound(ans.begin(), ans.end(), x);
        if (it == ans.end())    {
            ans.push_back(x);
        } 
        else {
            *it = x;
        }
    }
    return ans.size();
}


vector<int> compute_lps(string s) 
{ 
    int n = s.size(); 
  
    // To store longest prefix suffix 
    vector<int> lps(n); 
  
    // Length of the previous 
    // longest prefix suffix 
    int len = 0; 
  
    // lps[0] is always 0 
    lps[0] = 0; 
    int i = 1; 
  
    // Loop calculates lps[i] for i = 1 to n - 1 
    while (i < n) { 
        if (s[i] == s[len]) { 
            len++; 
            lps[i] = len; 
            i++; 
        } 
  
        // (pat[i] != pat[len]) 
        else { 
            if (len != 0) 
                len = lps[len - 1]; 
            // Also, note that we do not increment 
            // i here 
  
            // If len = 0 
            else { 
                lps[i] = 0; 
                i++; 
            } 
        } 
    } 
  
    return lps; 
} 


// new lps
vector<int> lps(string S) {
    vector<int> z(S.size());
    int l = -1, r = -1;
    for(int i=1; i < S.size(); ++i){
        z[i] = i >= r ? 0 : min(r - i, z[i - l]);
        while (i + z[i] < S.size() && S[i + z[i]] == S[z[i]])
            z[i]++;
        if (i + z[i] > r)
            l = i, r = i + z[i];
    }
    return z;
}

vector<int> parent(1e6 + 10,-1);
vector<int> conatiner(1e6 + 10,0);


int find_set(int v) {
    if (v == parent[v])
        return v;
    return parent[v] = find_set(parent[v]);
}

void make_set(int v) {
    parent[v] = v;
    conatiner[v] = 1;
}

void union_sets(int a, int b) {
    a = find_set(a);
    b = find_set(b);
    if (a != b) {
        if (conatiner[a] < conatiner[b])
            swap(a, b);
        parent[b] = a;
        conatiner[a] += conatiner[b];
    }
}

// void union_sets(int a, int b) {
//     a = find_set(a).first;
//     b = find_set(b).first;
//     if (a != b) {
//         if (rank[a] < rank[b])
//             swap(a, b);
//         parent[b] = make_pair(a, 1);
//         if (rank[a] == rank[b])
//             rank[a]++;
//     }
// }


// Sum queries in Segment Tree


int n, t[4*N];


void build(int a[], int v, int tl, int tr) {
    if (tl == tr) {
        t[v] = a[tl];
    } else {
        int tm = (tl + tr) / 2;
        build(a, v*2, tl, tm);
        build(a, v*2+1, tm+1, tr);
        t[v] = t[v*2] + t[v*2+1];
    }
}

int sum(int v, int tl, int tr, int l, int r) {
    if (l > r) 
        return 0;
    if (l == tl && r == tr) {
        return t[v];
    }
    int tm = (tl + tr) / 2;
    return sum(v*2, tl, tm, l, min(r, tm))
           + sum(v*2+1, tm+1, tr, max(l, tm+1), r);
}


void update(int v, int tl, int tr, int pos, int new_val) {
    if (tl == tr) {
        t[v] = new_val;
    } else {
        int tm = (tl + tr) / 2;
        if (pos <= tm)
            update(v*2, tl, tm, pos, new_val);
        else
            update(v*2+1, tm+1, tr, pos, new_val);
        t[v] = t[v*2] + t[v*2+1];
    }
}



// Lazy 


void build(int a[], int v, int tl, int tr) {
    if (tl == tr) {
        t[v] = a[tl];
    } else {
        int tm = (tl + tr) / 2;
        build(a, v*2, tl, tm);
        build(a, v*2+1, tm+1, tr);
        t[v] = 0;
    }
}

void update(int v, int tl, int tr, int l, int r, int add) {
    if (l > r)
        return;
    if (l == tl && r == tr) {
        t[v] += add;
    } else {
        int tm = (tl + tr) / 2;
        update(v*2, tl, tm, l, min(r, tm), add);
        update(v*2+1, tm+1, tr, max(l, tm+1), r, add);
    }
}

int get(int v, int tl, int tr, int pos) {
    if (tl == tr)
        return t[v];
    int tm = (tl + tr) / 2;
    if (pos <= tm)
        return t[v] + get(v*2, tl, tm, pos);
    else
        return t[v] + get(v*2+1, tm+1, tr, pos);
}


// RMQ

int minVal(int x, int y) { return (x < y)? x: y; }  
  
// A utility function to get the  
// middle index from corner indexes.  
int getMid(int s, int e) { return s + (e -s)/2; }  
  
/* A recursive function to get the 
minimum value in a given range  
of array indexes. The following  
are parameters for this function.  
  
    st --> Pointer to segment tree  
    index --> Index of current node in the  
           segment tree. Initially 0 is  
           passed as root is always at index 0  
    ss & se --> Starting and ending indexes  
                of the segment represented  
                by current node, i.e., st[index]  
    qs & qe --> Starting and ending indexes of query range */
int RMQUtil(int *st, int ss, int se, int qs, int qe, int index)  
{  
    // If segment of this node is a part  
    // of given range, then return  
    // the min of the segment  
    if (qs <= ss && qe >= se)  
        return st[index];  
  
    // If segment of this node 
    // is outside the given range  
    if (se < qs || ss > qe)  
        return INT_MAX;  
  
    // If a part of this segment 
    // overlaps with the given range  
    int mid = getMid(ss, se);  
    return minVal(RMQUtil(st, ss, mid, qs, qe, 2*index+1),  
                RMQUtil(st, mid+1, se, qs, qe, 2*index+2));  
}  
  
// Return minimum of elements in range 
// from index qs (query start) to  
// qe (query end). It mainly uses RMQUtil()  
int RMQ(int *st, int n, int qs, int qe)  
{  
    // Check for erroneous input values  
    if (qs < 0 || qe > n-1 || qs > qe)  
    {  
        cout<<"Invalid Input";  
        return -1;  
    }  
  
    return RMQUtil(st, 0, n-1, qs, qe, 0);  
}  
  
// A recursive function that constructs 
// Segment Tree for array[ss..se].  
// si is index of current node in segment tree st  
int constructSTUtil(int arr[], int ss, int se, 
                                int *st, int si)  
{  
    // If there is one element in array, 
    // store it in current node of  
    // segment tree and return  
    if (ss == se)  
    {  
        st[si] = arr[ss];  
        return arr[ss];  
    }  
  
    // If there are more than one elements,  
    // then recur for left and right subtrees  
    // and store the minimum of two values in this node  
    int mid = getMid(ss, se);  
    st[si] = minVal(constructSTUtil(arr, ss, mid, st, si*2+1),  
                    constructSTUtil(arr, mid+1, se, st, si*2+2));  
    return st[si];  
}  
  
/* Function to construct segment tree  
from given array. This function allocates 
memory for segment tree and calls constructSTUtil() to  
fill the allocated memory */
int *constructST(int arr[], int n)  
{  
    // Allocate memory for segment tree  
  
    //Height of segment tree  
    int x = (int)(ceil(log2(n)));  
  
    // Maximum size of segment tree  
    int max_size = 2*(int)pow(2, x) - 1;  
  
    int *st = new int[max_size];  
  
    // Fill the allocated memory st  
    constructSTUtil(arr, 0, n-1, st, 0);  
  
    // Return the constructed segment tree  
    return st;  
}  




// query l to r range for the no of integers between x and y


// #include <iostream>
// using namespace std;
// int T = 1;
// const int N = 1e6;
// const int MX = N;
// struct node{
// 	int l, r, cnt;	
// }t[100*MX];
// int root[N], a[N];
// int build(int lo, int hi){
// 	int id = T++;
// 	if(lo == hi) return id;
// 	int  mid = (lo+hi)/2;
// 	t[id].l = build(lo, mid);
// 	t[id].r = build(mid+1, hi);
// 	return id;
// }
// int update(int rt, int lo, int hi, int val){
// 	int id = T++;
// 	t[id] = t[rt]; t[id].cnt++;
// 	if(lo == hi) return id;
// 	int mid = (lo+hi)/2;
// 	if(val <= mid) t[id].l = update(t[rt].l, lo, mid, val);
// 	else t[id].r = update(t[rt].r, mid+1, hi, val);
// 	return id;
// }
// int query(int rt, int lo, int hi, int x, int y){
// 	if(x==lo and y==hi) return t[rt].cnt;
// 	int mid = (lo+hi)/2;
// 	if(y <= mid) return query(t[rt].l, lo, mid, x, y);
// 	else if (x > mid) return query(t[rt].r, mid+1, hi, x, y);
// 	return query(t[rt].l, lo, mid, x, mid)	+ query(t[rt].r, mid+1, hi, mid+1, y);
// }
// int main() {
// 	int i, n, q;
// 	cin >> n >> q;
// 	for(i = 0; i < n; i++) cin >> a[i+1];
// 	root[0] = build(0, MX);
// 	for(i = 1; i <= n; i++){
// 		root[i] = update(root[i-1], 0, MX, a[i]);
// 	}
// 	while(q--){
// 		int l, r, x, y;
// 		cin >> l >> r >> x >> y;
// 		cout << query(root[r], 0, MX, x, y) - query(root[l-1], 0, MX, x, y) << endl;
// 	}
// 	return 0;
// }







// tenplate for polciy based DS

#include <ext/pb_ds/assoc_container.hpp> // Common file 
using namespace __gnu_pbds; 
typedef tree<int, null_type, less<int>, rb_tree_tag, 
             tree_order_statistics_node_update> 
    new_data_set; 


typedef tree<pair<int,int>, null_type, less<pair<int,int>>, rb_tree_tag, 
             tree_order_statistics_node_update> 
    ordered_set; 





struct RollingHash{
	vector<int> pwr, hsh;
	int A, M;
	int n;

    RollingHash(){}
    
	RollingHash(string s, int _A = 31, int _M = 1e9 + 7){
		n = s.size();
		pwr.resize(n+1); hsh.resize(n+1);

		A = _A, M = _M;

		pwr[0] = 1;
		for(int i = 1; i <= n; i++) pwr[i] = pwr[i-1] * A % M;

		hsh[0] = s[0] % M + 1;
		for(int i = 1; i < n; i++){
			hsh[i] = (hsh[i - 1] * A % M) + s[i] + 1; if(hsh[i] >= M) hsh[i] -= M;
		}
	}

	int getHash(int x, int y){
		assert(x >= 0 and x < n and y >= 0 and y <= n);
		return (hsh[y] + M - ((x-1 >= 0)? hsh[x-1] * pwr[y-x+1] % M : 0)) % M;
	}
};

struct PalindromeChecker {
	RollingHash hash;
	RollingHash revHash;
    int n;

	PalindromeChecker(string s): hash(s), n(s.size()) {
        reverse(s.begin(), s.end());
        revHash = RollingHash(s);
    }

    bool isPalindrome(int i, int j) {
        return hash.getHash(i, j) == revHash.getHash(n-j-1, n-i-1);
    }

};

// COMBINATIONS

const int maxn = 109;
int C[maxn + 1][maxn + 1];

void COM() {
        C[0][0] = 1;
        for (int n = 1; n <= maxn; ++n) {
        C[n][0] = C[n][n] = 1;
        for (int k = 1; k < n; ++k)
                C[n][k] = C[n - 1][k - 1] + C[n - 1][k];
        }
}

int n; // number of vertices
vector<vector<int>> adj; // adjacency list of graph
vector<bool> visited;
vector<int> ans;

void dfs(int v) {
    visited[v] = true;
    for (int u : adj[v]) {
        if (!visited[u])
            dfs(u);
    }
    ans.push_back(v);
}

void topological_sort() {
    visited.assign(n, false);
    ans.clear();
    for (int i = 0; i < n; ++i) {
        if (!visited[i])
            dfs(i);
    }
    reverse(ans.begin(), ans.end());
}
