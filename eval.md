### Problem Statement

Takahashi wants to set his password for a certain website to a string $P$ consisting of lowercase English letters.  
The password for that website must be a string of length at least $L$.

Determine whether $P$ satisfies the length condition, that is, whether it is a string of length at least $L$.
### Constraints

-   $P$ is a string consisting of lowercase English letters with length between $1$ and $100$, inclusive.
-   $1 \leq L \leq 100$
-   $L$ is an integer.
### Input

The input is given from Standard Input in the following format:

```
$P$
$L$
```
### Output

If $P$ satisfies the length condition, print `Yes`; otherwise, print `No`.
Solve this using Python

### Problem Statement

There are $N$ squares arranged in a row from left to right. Initially, all squares are painted white.

Process $Q$ queries in order. The $i$\-th query gives an integer $A_i$ between $1$ and $N$, inclusive, and performs the following operation:

> Flip the color of the $A_i$\-th square from the left. Specifically, if the $A_i$\-th square from the left is painted white, paint it black; if it is painted black, paint it white.  
> Then, find the number of intervals of consecutively painted black squares.
> 
> Here, an interval of consecutively painted black squares is a pair of integers $(l,r)$ $(1\leq l\leq r\leq N)$ that satisfy all of the following:
> 
> -   The $l$\-th, $(l+1)$\-th, $\ldots$, $r$\-th squares from the left are all painted black.
> -   Either $l=1$, or the $(l-1)$\-th square from the left is painted white.
> -   Either $r=N$, or the $(r+1)$\-th square from the left is painted white.
### Constraints

-   $1\leq N,Q\leq 5\times 10^5$
-   $1\leq A_i\leq N$
-   All input values are integers.
### Input

The input is given from Standard Input in the following format:

```
$N$ $Q$
$A_1$ $A_2$ $\ldots$ $A_Q$
```
### Output

Output $Q$ lines. On the $i$\-th line $(1\leq i\leq Q)$, output the answer to the $i$\-th query.
### Sample Input 1

```
5 7
2 3 3 5 1 5 2
```
### Sample Output 1

```
1
1
1
2
2
1
1
```

Below, the $i$\-th square from the left is referred to as square $i$.  
After each query, the state is as follows:

-   After the $1$st query, only square $2$ is painted black. There is $1$ interval of consecutively painted black squares: $(l,r)=(2,2)$.
-   After the $2$nd query, squares $2,3$ are painted black. There is $1$ interval of consecutively painted black squares: $(l,r)=(2,3)$.
-   After the $3$rd query, only square $2$ is painted black. There is $1$ interval of consecutively painted black squares: $(l,r)=(2,2)$.
-   After the $4$th query, squares $2,5$ are painted black. There are $2$ intervals of consecutively painted black squares: $(l,r)=(2,2), (5,5)$.
-   After the $5$th query, squares $1,2,5$ are painted black. There are $2$ intervals of consecutively painted black squares: $(l,r)=(1,2), (5,5)$.
-   After the $6$th query, only squares $1,2$ are painted black. There is $1$ interval of consecutively painted black squares: $(l,r)=(1,2)$.
-   After the $7$th query, only square $1$ is painted black. There is $1$ interval of consecutively painted black squares: $(l,r)=(1,1)$.

Thus, output $1,1,1,2,2,1,1$ separated by newlines.
Solve this using Python