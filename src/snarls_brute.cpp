#include <fstream>
#include <iostream>
#include <set>
#include <vector>

using namespace std;

#define MAX_N 10000

vector<pair<int, string>> adj[MAX_N];
bool separable[MAX_N][MAX_N][2][2]; 
set<int> nodes;

int REACH_FLAG = 1;
int BAD_FLAG = 2;

int dfs_it = 1;
int seen[MAX_N];

vector<int> snarl_component;

// traverse {xd_x, yd_y} with dfs (currently at vertex z) to test separability or to compute snarl component
int dfs(int z, int x, int y, char dx, char dy, bool track_component) {
	if (seen[z] == dfs_it) return 0;
	if (track_component) {
		snarl_component.push_back(z);
	}
	seen[z] = dfs_it;
	int ans = 0;
	if (z == x) {
		for (auto incidence : adj[z]) {
			if (incidence.second[0] != dx) continue;
			if (incidence.first == y && incidence.second[1] == dy) ans |= REACH_FLAG;
			if (incidence.first == y && incidence.second[1] != dy) ans |= BAD_FLAG; 
			ans |= dfs(incidence.first, x, y, dx, dy, track_component);
		}
	} else if (z == y) {
		for (auto incidence : adj[z]) {
			if (incidence.second[0] != dy) continue;
			if (incidence.first == x && incidence.second[1] != dx) ans |= BAD_FLAG; 
			ans |= dfs(incidence.first, x, y, dx, dy, track_component);
		}
	} else {
		for (auto incidence : adj[z]) {
			if (incidence.first == y && incidence.second[1] == dy) ans |= REACH_FLAG;
			if (incidence.first == y && incidence.second[1] != dy) ans |= BAD_FLAG; 
			if (incidence.first == x && incidence.second[1] != dx) ans |= BAD_FLAG; 
			ans |= dfs(incidence.first, x, y, dx, dy, track_component);
		}
	}
	return ans;
}

bool test_separability(int x, int y, char dx, char dy) {
	int out = dfs(x, x, y, dx, dy, false);
	dfs_it++;
	return (REACH_FLAG & out) && !(BAD_FLAG & out);
}

void compute_component(int x, int y, char dx, char dy) {
	dfs(x, x, y, dx, dy, true);
	dfs_it++;
}

int main(int argc, char** argv) {
	if (argc != 2) {
		cerr<<"Usage: ./snarls_brute input_file"<<endl;
		return 1;
	}
	cerr<<"Reading input"<<endl;
	ifstream fin(argv[1]);
	while (true) {
		string type;
		fin>>type;
		if (type == "S") {
			int dx;
			fin>>dx;
			nodes.insert(dx);
			fin>>type;
		} else if (type == "L") {
			int x, y;
			string dx, dy;
			fin>>x>>dx>>y>>dy>>type;
			if (dy == "-") dy = "+";
			else dy = "-";
			adj[x].push_back({y, dx + dy});
			adj[y].push_back({x, dy + dx});
		}
		if (fin.eof()) break;
	}
	cerr<<"Input read"<<endl;

	fin.close();
	int tips = 0;
	for (int node : nodes) {
		bool plus_incidence = false;
		bool minus_incidence = false;
		for (auto incidence : adj[node]) {
			if (incidence.second[0] == '+') plus_incidence = true;
			else minus_incidence = true;
		}
		if (!plus_incidence || !minus_incidence) tips++;
	}
	cerr<<"Vertices: "<<nodes.size()<<endl;
	cerr<<"Tips: "<<tips<<endl;

	cerr<<"Computing separabilities"<<endl;
	for (int x : nodes) {
		for (int y : nodes) {
			if (y <= x) continue;
			separable[y][x][0][0] = separable[x][y][0][0] = test_separability(x, y, '+', '+');
			separable[y][x][1][0] = separable[x][y][0][1] = test_separability(x, y, '+', '-');
			separable[y][x][0][1] = separable[x][y][1][0] = test_separability(x, y, '-', '+');
			separable[y][x][1][1] = separable[x][y][1][1] = test_separability(x, y, '-', '-');
		}
	}

	cerr<<"Computing minimalities"<<endl;
	for (int x : nodes) {
		for (int y : nodes) {
			if (y <= x) continue;
			for (int dx = 0; dx < 2; dx++) {
				for (int dy = 0; dy < 2; dy++) {
					if (!separable[x][y][dx][dy]) continue;
					snarl_component.clear();
					compute_component(x, y, dx ? '-' : '+', dy ? '-' : '+');
					for (int z : snarl_component) {
						if (z == x || z == y) continue;
						if (separable[x][z][dx][0] && separable[z][y][1][dy]) {
							goto end;
						}
						if (separable[x][z][dx][1] && separable[z][y][0][dy]) {
							goto end;
						}
					}
					cout<<x;
					if (dx) cout<<"-";
					else cout<<"+";
					cout<<" ";
					cout<<y;
					if (dy) cout<<"-";
					else cout<<"+";
					cout<<endl;
					end:;
				}
			}
		}
	}
}