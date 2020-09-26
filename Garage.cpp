#include<bits/stdc++.h>
#define ll long long
#define MOD 1000000007
#define mod 1000000007
using namespace std;




bool isprime(ll n) //Time Complexity--->sqrt(n)
{   if (n <= 1)  return false; if (n <= 3)  return true; 
  	if (n%2 == 0 || n%3 == 0) return false; 
    for (int i=5; i*i<=n; i=i+6) if (n%i == 0 || n%(i+2) == 0) return false; 
    return true; 
} 


// a power to b mod m
// (a^b) % m

long long binpow(long long a, long long b, long long m) {
    a %= m;
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
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

bool find_any_solution(int a, int b, int c, int &x0, int &y0, int &g) {
    g = gcd(abs(a), abs(b), x0, y0);
    if (c % g) {
        return false;
    }

    x0 *= c / g;
    y0 *= c / g;
    if (a < 0) x0 = -x0;
    if (b < 0) y0 = -y0;
    return true;
}


// Find all the solution of ax + by = c with min and max


void shift_solution(int & x, int & y, int a, int b, int cnt) {
    x += cnt * b;
    y -= cnt * a;
}

int find_all_solutions(int a, int b, int c, int minx, int maxx, int miny, int maxy) {
    int x, y, g;
    if (!find_any_solution(a, b, c, x, y, g))
        return 0;
    a /= g;
    b /= g;

    int sign_a = a > 0 ? +1 : -1;
    int sign_b = b > 0 ? +1 : -1;

    shift_solution(x, y, a, b, (minx - x) / b);
    if (x < minx)
        shift_solution(x, y, a, b, sign_b);
    if (x > maxx)
        return 0;
    int lx1 = x;

    shift_solution(x, y, a, b, (maxx - x) / b);
    if (x > maxx)
        shift_solution(x, y, a, b, -sign_b);
    int rx1 = x;

    shift_solution(x, y, a, b, -(miny - y) / a);
    if (y < miny)
        shift_solution(x, y, a, b, -sign_a);
    if (y > maxy)
        return 0;
    int lx2 = x;

    shift_solution(x, y, a, b, -(maxy - y) / a);
    if (y > maxy)
        shift_solution(x, y, a, b, sign_a);
    int rx2 = x;

    if (lx2 > rx2)
        swap(lx2, rx2);
    int lx = max(lx1, lx2);
    int rx = min(rx1, rx2);

    if (lx > rx)
        return 0;
    return (rx - lx) / abs(b) + 1;
}


bool isVowel(char c) {
    if(c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U') {
        return true;
    }
    return false;
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
    fac[i] = fac[i - 1] * i % MOD;
    inv[i] = MOD - inv[MOD%i] * (MOD / i) % MOD;
    finv[i] = finv[i - 1] * inv[i] % MOD;
  }
}
 
 
 
long long COM(int n, int k){
  if (n < k) return 0;
  if (n < 0 || k < 0) return 0;
  return fac[n] * (finv[k] * finv[n - k] % MOD) % MOD;
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
  
    // stores the smallest prime number 
    // that divides n 
    int dup = factor[n]; 
  
    // stores the count of number of times 
    // a prime number divides n. 
    int c = 1; 
  
    // reduces to the next number after prime 
    // factorization of n 
    int j = n / factor[n]; 
  
    // false when prime factorization is done 
    while (j != 1) { 
        // if the same prime number is dividing n, 
        // then we increase the count 
        if (factor[j] == dup) 
            c += 1; 
  
        /* if its a new prime factor that is factorizing n,  
           then we again set c=1 and change dup to the new  
           prime factor, and apply the formula explained  
           above. */
        else { 
            dup = factor[j]; 
            ans = ans * (c + 1); 
            c = 1; 
        } 
  
        // prime factorizes a number 
        j = j / factor[j]; 
    } 
  
    // for the last prime factor 
    ans = ans * (c + 1); 
  
    return ans; 
} 
 
