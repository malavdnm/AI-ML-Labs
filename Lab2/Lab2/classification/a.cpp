#include <bits/stdc++.h>
using namespace std;
int main(){
//		   w-------x
//		 / |     / |
//		a------b   |
//		|  |   |   |
//		| y--- |---z
//		|/     |  /
//		c------d
//
	int n;
	cin>>n;
	int count=0;
	if(n==2){
//    a------b
//	 /      /
//	c------d

		int a,b,c,d;
		count=16;
		bool bad;
		for(int i=0; i<16; i++){
			a=i%2;
			b=i%4/2;
			c=i%8/4;
			d=i/8;
			bad=false;
			if(a==d && b==c && a!=b)
				bad=true;
			if(bad) count--;
		}
	}
	if(n==3){
		int a,b,c,d,w,x,y,z;
		count=256;
		bool bad;
		for(int i=0; i<256; i++){
			z=i%2;
			y=i%4/2;
			x=i%8/4;
			w=i%16/8;
			d=i%32/16;
			c=i%64/32;
			b=i%128/64;
			a=i/128;
			bad=false;
			if(a==d && b==c && a!=b)
			bad=true;
			if(a==y && w==c && a!=w)
			bad=true;
			if(w==z && x==y && w!=x)
			bad=true;
			if(x==d && b==z && x!=b)
			bad=true;
			if(a==x && b==w && a!=w)
			bad=true;
			if(y==d && z==c && d!=z)
			bad=true;
			if(a==z && c==x && a!=x)
			bad=true;
			if(b==y && d==w && b!=w)
			bad=true;
			if(a==z && b==y && a!=y)
			bad=true;
			if(c==x && d==w && c!=w)
			bad=true;
			if(a==z && w==d && a!=w)
			bad=true;
			if(c==x && b==y && c!=y)
			bad=true;
			if(bad)
			count--;
		}
	}
	cout<<count<<endl;
}