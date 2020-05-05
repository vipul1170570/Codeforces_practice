#include<bits/stdc++.h>
#include<unordered_set>
#define initdp(a,b) for(int i=0;i<=a;i++)for(int j=0;j<=b;j++)dp[i][j]=-1;
#define fi first
#define se second
#define pb push_back
#define pii pair<int,int>
#define ll long long
#define pll pair<ll,ll>
#define all(arr) arr.begin(), arr.end()
#define rep(i,n) for(int i=0;i<n;i++)
#define repd(i,n) for(int i=n-1;i>=0;i--)
#define fo(i,l,r) for(int i=l;i<=r;i++)
#define inf 1000000001
#define inf1 1000000000000000001
#define mod 1000000007
#define pie 3.14159265358979323846
#define N 1000005
#define mid(l,r) l+(r-l)/2
#define vec vector<int>
#define yes cout<<"YES"<<endl;
#define no cout<<"NO"<<endl;
#define end cout<<endl;
using namespace std;

bool visited[N];
int child[N], depth[N];
vector <int> adj[N];

inline int gcd(int a, int b) {
	while (a != 0 && b != 0) {
		if (a > b) {
			a %= b;
		} else {
			b %= a;
		}
	}
	return max(a, b);
}

void dfs(int u) {
	visited[u]=true;
	child[u]=0;
	for(auto i: adj[u]) {
		if(!visited[i]) {
			depth[i] = depth[u]+1;
			dfs(i);
			child[u] += (1+child[i]);
		}
	}
}

void solve()
{
    
}
int main()
{
	ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);
    int t=1;
    cin>>t;
    while(t--)
    {
        
    }
}