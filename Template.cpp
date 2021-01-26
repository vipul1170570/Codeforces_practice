#include "bits/stdc++.h"
#define pb push_back
#define mpr make_pair
#define pii pair<int, int>
#define ll long long
#define dl long double
#define all(arr) arr.begin(), arr.end()
#define fi first
#define se second
#define rep(i, n) for (int i = 0; i < (int)(n); i++)
#define fo(i, l, r) for(int i=l; i <= r; i++)
#define pie 3.14159265358979323846264338327950L
#define mid(l, r) l + (r - l) / 2
#define memo(arr) memset(arr, 0,sizeof(arr));
#define mem1(arr) memset(arr,1,sizeof(arr));
#define vi vector<int>
#define vs vector<string>
#define str string
#define vvi vector<vector<int>>
#define vvpi vector<vector<pii>>
#define mii map<int,int>
#define mci map<char,int>
#define si set<int>
#define yes cout << "YES" << endl;
#define no cout << "NO" << endl;
#define endl "\n"
#define int long long
#define sz(x) (int)x.size()
#define template_array_size (int)1e6 + 6
#define INF (int)1e18
#define ret return
#define trace1(x)                cerr << #x << ": " << x << endl;
#define trace2(x, y)             cerr << #x << ": " << x << " | " << #y << ": " << y << endl;
#define trace3(x, y, z)          cerr << #x << ": " << x << " | " << #y << ": " << y << " | " << #z << ": " << z << endl;
#define trace4(a, b, c, d)       cerr << #a << ": " << a << " | " << #b << ": " << b << " | " << #c << ": " << c << " | " << #d << ": " << d << endl;
#define trace5(a, b, c, d, e)    cerr << #a << ": " << a << " | " << #b << ": " << b << " | " << #c << ": " << c << " | " << #d << ": " << d << " | " << #e << ": " << e << endl;
#define trace6(a, b, c, d, e, f) cerr << #a << ": " << a << " | " << #b << ": " << b << " | " << #c << ": " << c << " | " << #d << ": " << d << " | " << #e << ": " << e << " | " << #f << ": " << f << endl;
#define trace7(arr)              for(auto it : arr) cerr<<it<<" "; cerr<<endl;
using namespace std; 



int dx[4]={1,0,-1,0},dy[4]={0,1,0,-1};
int ddx[8]={1,1,0,-1,-1,-1,0,1},ddy[8]={0,1,1,1,0,-1,-1,-1};
int gcd(int a,int b){ if(!a)return b;return gcd(b%a,a);}
int lcm(int a, int b) { return (a*b)/ gcd(a,b);}
template <typename T> void ckmin(T &a, const T &b) { a = min(a, b); }
template <typename T> void ckmax(T &a, const T &b) { a = max(a, b); }


namespace __input {
    template <class T1, class T2> void re(pair<T1, T2> &p);
    template <class T> void re(vector<T> &a);
    template <class T, size_t SZ> void re(array<T, SZ> &a);
 
    template <class T> void re(T &x) { cin >> x; }
    void re(double &x) { string t; re(t); x = stod(t); }
    template <class Arg, class... Args> void re(Arg &first, Args &...rest) { re(first); re(rest...); }
 
    template <class T1, class T2> void re(pair<T1, T2> &p) { re(p.f, p.s); }
    template <class T> void re(vector<T> &a) { for (int32_t i = 0; i < sz(a); i++) re(a[i]); }
    template <class T, size_t SZ> void re(array<T, SZ> &a) { for (int32_t i = 0; i < SZ; i++) re(a[i]); }
}
using namespace __input;




const int MAX = 1e6 + 2047;
const int mod = 1e9 + 7;
const int MAX_2 = 1009;

void sout(){
    cout<<endl;
}
template <typename T,typename... Types>
void sout(T var1,Types... var2){
    cout<<var1<<" ";
    sout(var2...);
}
template<typename T>
void sout(vector<T> arr) {
    for(auto it : arr) {
        cout<<it<<" ";
    }
    sout();
}

template<typename T>
void sout(set<T> st) {
    for(auto it : st) {
        cout<<it<<" ";
    }
    sout();
}




//____________________________________________








void test_case (int tc = 0) {










}






int32_t main ()
{
    ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
    cout << fixed << setprecision(10);


    #ifndef ONLINE_JUDGE    
        // freopen("input.txt","r",stdin);
        // freopen("output.txt","w",stdout); 
    #endif

    int t = 1;
    // cin >> t;
    for(int i=1; i<=t; i++) {
            test_case(i);
    }

    // cerr << "\nTime elapsed: " << 1000 * clock() / CLOCKS_PER_SEC << "ms\n";
}


/* stuff you should look for
    * int overflow, array bounds
    * special cases (n=1?)
    * do smth instead of nothing and stay organized
    * WRITE STUFF DOWN
    * DON'T GET STUCK ON ONE APPROACH
*/

/* run in terminal

 g++ -std=c++14 -o a actionKamen.cpp && ./a

*/
