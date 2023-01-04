---
layout: single
title:  "Digit Recognition Implementation from Scratch(NeuralNetwork)"
categories: coding
tag: [python, ML, DL, kaggle]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>



```python
#
 
T
h
i
s
 
P
y
t
h
o
n
 
3
 
e
n
v
i
r
o
n
m
e
n
t
 
c
o
m
e
s
 
w
i
t
h
 
m
a
n
y
 
h
e
l
p
f
u
l
 
a
n
a
l
y
t
i
c
s
 
l
i
b
r
a
r
i
e
s
 
i
n
s
t
a
l
l
e
d

#
 
I
t
 
i
s
 
d
e
f
i
n
e
d
 
b
y
 
t
h
e
 
k
a
g
g
l
e
/
p
y
t
h
o
n
 
D
o
c
k
e
r
 
i
m
a
g
e
:
 
h
t
t
p
s
:
/
/
g
i
t
h
u
b
.
c
o
m
/
k
a
g
g
l
e
/
d
o
c
k
e
r
-
p
y
t
h
o
n

#
 
F
o
r
 
e
x
a
m
p
l
e
,
 
h
e
r
e
'
s
 
s
e
v
e
r
a
l
 
h
e
l
p
f
u
l
 
p
a
c
k
a
g
e
s
 
t
o
 
l
o
a
d


i
m
p
o
r
t
 
n
u
m
p
y
 
a
s
 
n
p
 
#
 
l
i
n
e
a
r
 
a
l
g
e
b
r
a

i
m
p
o
r
t
 
p
a
n
d
a
s
 
a
s
 
p
d
 
#
 
d
a
t
a
 
p
r
o
c
e
s
s
i
n
g
,
 
C
S
V
 
f
i
l
e
 
I
/
O
 
(
e
.
g
.
 
p
d
.
r
e
a
d
_
c
s
v
)

f
r
o
m
 
m
a
t
p
l
o
t
l
i
b
 
i
m
p
o
r
t
 
p
y
p
l
o
t
 
a
s
 
p
l
t


#
 
I
n
p
u
t
 
d
a
t
a
 
f
i
l
e
s
 
a
r
e
 
a
v
a
i
l
a
b
l
e
 
i
n
 
t
h
e
 
r
e
a
d
-
o
n
l
y
 
"
.
.
/
i
n
p
u
t
/
"
 
d
i
r
e
c
t
o
r
y

#
 
F
o
r
 
e
x
a
m
p
l
e
,
 
r
u
n
n
i
n
g
 
t
h
i
s
 
(
b
y
 
c
l
i
c
k
i
n
g
 
r
u
n
 
o
r
 
p
r
e
s
s
i
n
g
 
S
h
i
f
t
+
E
n
t
e
r
)
 
w
i
l
l
 
l
i
s
t
 
a
l
l
 
f
i
l
e
s
 
u
n
d
e
r
 
t
h
e
 
i
n
p
u
t
 
d
i
r
e
c
t
o
r
y


i
m
p
o
r
t
 
o
s

f
o
r
 
d
i
r
n
a
m
e
,
 
_
,
 
f
i
l
e
n
a
m
e
s
 
i
n
 
o
s
.
w
a
l
k
(
'
/
k
a
g
g
l
e
/
i
n
p
u
t
'
)
:

 
 
 
 
f
o
r
 
f
i
l
e
n
a
m
e
 
i
n
 
f
i
l
e
n
a
m
e
s
:

 
 
 
 
 
 
 
 
p
r
i
n
t
(
o
s
.
p
a
t
h
.
j
o
i
n
(
d
i
r
n
a
m
e
,
 
f
i
l
e
n
a
m
e
)
)


#
 
Y
o
u
 
c
a
n
 
w
r
i
t
e
 
u
p
 
t
o
 
2
0
G
B
 
t
o
 
t
h
e
 
c
u
r
r
e
n
t
 
d
i
r
e
c
t
o
r
y
 
(
/
k
a
g
g
l
e
/
w
o
r
k
i
n
g
/
)
 
t
h
a
t
 
g
e
t
s
 
p
r
e
s
e
r
v
e
d
 
a
s
 
o
u
t
p
u
t
 
w
h
e
n
 
y
o
u
 
c
r
e
a
t
e
 
a
 
v
e
r
s
i
o
n
 
u
s
i
n
g
 
"
S
a
v
e
 
&
 
R
u
n
 
A
l
l
"
 

#
 
Y
o
u
 
c
a
n
 
a
l
s
o
 
w
r
i
t
e
 
t
e
m
p
o
r
a
r
y
 
f
i
l
e
s
 
t
o
 
/
k
a
g
g
l
e
/
t
e
m
p
/
,
 
b
u
t
 
t
h
e
y
 
w
o
n
'
t
 
b
e
 
s
a
v
e
d
 
o
u
t
s
i
d
e
 
o
f
 
t
h
e
 
c
u
r
r
e
n
t
 
s
e
s
s
i
o
n
```

<pre>
/
k
a
g
g
l
e
/
i
n
p
u
t
/
d
i
g
i
t
-
r
e
c
o
g
n
i
z
e
r
/
s
a
m
p
l
e
_
s
u
b
m
i
s
s
i
o
n
.
c
s
v

/
k
a
g
g
l
e
/
i
n
p
u
t
/
d
i
g
i
t
-
r
e
c
o
g
n
i
z
e
r
/
t
r
a
i
n
.
c
s
v

/
k
a
g
g
l
e
/
i
n
p
u
t
/
d
i
g
i
t
-
r
e
c
o
g
n
i
z
e
r
/
t
e
s
t
.
c
s
v

</pre>
#
 
D
A
T
A
 
I
N
P
U
T
 
&
 
S
P
L
I
T



```python
d
e
f
 
o
n
e
_
h
o
t
(
Y
)
:

 
 
 
 
o
n
e
_
h
o
t
_
Y
 
=
 
n
p
.
z
e
r
o
s
(
(
Y
.
s
i
z
e
,
Y
.
m
a
x
(
)
+
1
)
)

 
 
 
 
o
n
e
_
h
o
t
_
Y
[
n
p
.
a
r
a
n
g
e
(
Y
.
s
i
z
e
)
,
Y
]
 
=
 
1

 
 
 
 
o
n
e
_
h
o
t
_
Y
 
=
 
o
n
e
_
h
o
t
_
Y
.
T

 
 
 
 
r
e
t
u
r
n
 
o
n
e
_
h
o
t
_
Y
```


```python
n
p
.
z
e
r
o
s
(
(
Y
_
t
r
a
i
n
.
s
i
z
e
,
 
Y
_
t
r
a
i
n
.
m
a
x
(
)
+
1
)
)
.
s
h
a
p
e
```

<pre>
(
4
2
0
0
0
,
 
1
0
)
</pre>

```python
d
a
t
a
 
=
 
p
d
.
r
e
a
d
_
c
s
v
(
'
/
k
a
g
g
l
e
/
i
n
p
u
t
/
d
i
g
i
t
-
r
e
c
o
g
n
i
z
e
r
/
t
r
a
i
n
.
c
s
v
'
)

d
a
t
a
1
 
=
 
d
a
t
a

d
a
t
a
 
=
 
n
p
.
a
r
r
a
y
(
d
a
t
a
)

m
,
 
n
 
=
 
d
a
t
a
.
s
h
a
p
e

n
p
.
r
a
n
d
o
m
.
s
h
u
f
f
l
e
(
d
a
t
a
)



d
a
t
a
_
t
r
a
i
n
 
=
 
d
a
t
a
[
:
m
]
.
T

Y
_
t
r
a
i
n
 
=
 
d
a
t
a
_
t
r
a
i
n
[
0
]

X
_
t
r
a
i
n
 
=
 
d
a
t
a
_
t
r
a
i
n
[
1
:
n
]

X
_
t
r
a
i
n
 
=
 
X
_
t
r
a
i
n
 
/
 
2
5
5
.

#
_
,
m
_
t
r
a
i
n
 
=
 
X
_
t
r
a
i
n
.
s
h
a
p
e

```

#
 
F
o
r
w
a
r
d
 
p
r
o
p
a
g
a
t
i
o
n




$
Z
^
{
[
1
]
}
 
=
 
W
^
{
[
1
]
}
 
X
 
+
 
b
^
{
[
1
]
}
$




$
A
^
{
[
1
]
}
 
=
 
R
e
L
U
(
Z
^
{
[
1
]
}
)
)
$




$
Z
^
{
[
2
]
}
 
=
 
W
^
{
[
2
]
}
 
A
^
{
[
1
]
}
 
+
 
b
^
{
[
2
]
}
$




$
A
^
{
[
2
]
}
 
=
 
s
o
f
t
m
a
x
(
Z
^
{
[
2
]
}
)
$


#
 
B
a
c
k
w
a
r
d
 
p
r
o
p
a
g
a
t
i
o
n




$
L
o
s
s
 
=
 
-
 
\
S
i
g
m
a
[
{
y
_
{
[
k
]
}
 
*
l
o
g
(
a
_
{
[
k
]
}
)
]
}
$




$
C
o
s
t
 
=
 
-
 
\
f
r
a
c
{
1
}
{
m
}
\
S
i
g
m
a
\
S
i
g
m
a
[
{
y
_
{
[
k
]
}
 
*
l
o
g
(
a
_
{
[
k
]
}
)
]
}
$




$
d
Z
^
{
[
2
]
}
 
=
 
A
^
{
[
2
]
}
 
-
 
Y
$
 
 
:
 
D
e
r
i
v
a
t
i
v
e
 
o
f
 
L
o
s
s
 
F
u
n
c
t
i
o
n
 
w
i
t
h
 
r
e
s
p
e
c
t
 
t
o
 
Z




$
d
W
^
{
[
2
]
}
 
=
 
\
f
r
a
c
{
1
}
{
m
}
 
d
Z
^
{
[
2
]
}
 
A
^
{
[
1
]
T
}
$
 
:
 
D
e
r
i
v
a
t
i
v
e
 
o
f
 
C
o
s
t
 
F
u
n
c
t
i
o
n
 
w
i
t
h
 
r
e
s
p
e
c
t
 
t
o
 
W




$
d
B
^
{
[
2
]
}
 
=
 
\
f
r
a
c
{
1
}
{
m
}
 
\
S
i
g
m
a
 
{
d
Z
^
{
[
2
]
}
}
$




$
d
Z
^
{
[
1
]
}
 
=
 
W
^
{
[
2
]
T
}
 
d
Z
^
{
[
2
]
}
 
.
*
 
g
^
{
[
1
]
\
p
r
i
m
e
}
 
(
z
^
{
[
1
]
}
)
$




$
d
W
^
{
[
1
]
}
 
=
 
\
f
r
a
c
{
1
}
{
m
}
 
d
Z
^
{
[
1
]
}
 
A
^
{
[
0
]
T
}
$




$
d
B
^
{
[
1
]
}
 
=
 
\
f
r
a
c
{
1
}
{
m
}
 
\
S
i
g
m
a
 
{
d
Z
^
{
[
1
]
}
}
$




#
 
U
p
d
a
t
i
n
g
 
P
a
r
a
m
e
t
e
r
s




$
a
l
p
h
a
 
 
:
 
l
e
a
r
n
i
n
g
 
r
a
t
e
$




$
W
^
{
[
2
]
}
 
:
=
 
W
^
{
[
2
]
}
 
-
 
\
a
l
p
h
a
 
d
W
^
{
[
2
]
}
$




$
b
^
{
[
2
]
}
 
:
=
 
b
^
{
[
2
]
}
 
-
 
\
a
l
p
h
a
 
d
b
^
{
[
2
]
}
$




$
W
^
{
[
1
]
}
 
:
=
 
W
^
{
[
1
]
}
 
-
 
\
a
l
p
h
a
 
d
W
^
{
[
1
]
}
$




$
b
^
{
[
1
]
}
 
:
=
 
b
^
{
[
1
]
}
 
-
 
\
a
l
p
h
a
 
d
b
^
{
[
1
]
}
$



```python
d
e
f
 
i
n
i
t
_
p
a
r
a
m
s
(
)
:

 
 
 
 
W
1
 
=
 
n
p
.
r
a
n
d
o
m
.
r
a
n
d
(
1
0
,
7
8
4
)
 
-
 
0
.
5

 
 
 
 
b
1
 
=
 
n
p
.
r
a
n
d
o
m
.
r
a
n
d
(
1
0
,
1
)
 
-
 
0
.
5

 
 
 
 
W
2
 
=
 
n
p
.
r
a
n
d
o
m
.
r
a
n
d
(
1
0
,
1
0
)
 
-
 
0
.
5

 
 
 
 
b
2
 
=
 
n
p
.
r
a
n
d
o
m
.
r
a
n
d
(
1
0
,
1
)
 
-
 
0
.
5

 
 
 
 
r
e
t
u
r
n
 
W
1
,
b
1
,
W
2
,
b
2


d
e
f
 
R
e
L
U
(
Z
)
:

 
 
 
 
r
e
t
u
r
n
 
n
p
.
m
a
x
i
m
u
m
(
Z
,
0
)


d
e
f
 
s
o
f
t
m
a
x
(
Z
)
:

 
 
 
 
r
e
t
u
r
n
 
n
p
.
e
x
p
(
Z
)
/
s
u
m
(
n
p
.
e
x
p
(
Z
)
,
 
0
)

 
 
 
 


d
e
f
 
f
o
r
w
a
r
d
_
p
r
o
p
(
W
1
,
b
1
,
W
2
,
b
2
,
X
)
:

 
 
 
 
Z
1
 
=
 
W
1
.
d
o
t
(
X
)
 
+
 
b
1

 
 
 
 
A
1
 
=
 
R
e
L
U
(
Z
1
)

 
 
 
 
Z
2
 
=
 
W
2
.
d
o
t
(
A
1
)
 
+
 
b
2

 
 
 
 
A
2
 
=
 
s
o
f
t
m
a
x
(
Z
2
)

 
 
 
 
r
e
t
u
r
n
 
Z
1
,
A
1
,
Z
2
,
A
2


#
c
o
s
t
 
f
u
n
c
t
i
o
n

d
e
f
 
c
o
s
t
_
f
u
n
c
t
i
o
n
(
A
2
,
o
n
e
_
h
o
t
_
Y
)
:

 
 
 
 
m
 
=
 
Y
.
s
h
a
p
e
[
1
]

 
 
 
 
c
o
s
t
 
=
 
-
(
1
/
m
)
*
n
p
.
s
u
m
(
Y
*
n
p
.
l
o
g
(
A
2
)
)

 
 
 
 
r
e
t
u
r
n
 
c
o
s
t



d
e
f
 
o
n
e
_
h
o
t
(
Y
)
:

 
 
 
 
o
n
e
_
h
o
t
_
Y
 
=
 
n
p
.
z
e
r
o
s
(
(
Y
.
s
i
z
e
,
Y
.
m
a
x
(
)
+
1
)
)

 
 
 
 
o
n
e
_
h
o
t
_
Y
[
n
p
.
a
r
a
n
g
e
(
Y
.
s
i
z
e
)
,
Y
]
 
=
 
1
 
 
 
 
 
 
 
#
 
f
o
r
 
e
v
e
r
y
 
o
b
s
e
r
v
a
t
i
o
n
(
r
o
w
)
,
 
a
c
c
e
s
s
 
t
o
 
t
h
e
 
c
o
l
u
m
n
 
s
p
e
c
i
f
i
e
d
 
b
y
 
t
h
e
 
l
a
b
e
l
 
i
n
 
Y

 
 
 
 
o
n
e
_
h
o
t
_
Y
 
=
 
o
n
e
_
h
o
t
_
Y
.
T

 
 
 
 
r
e
t
u
r
n
 
o
n
e
_
h
o
t
_
Y

 
 
 
 


d
e
f
 
d
e
r
i
v
_
R
e
L
U
(
Z
)
:

 
 
 
 
r
e
t
u
r
n
 
Z
>
0

 
 
 
 

 
 
 
 

d
e
f
 
b
a
c
k
_
p
r
o
p
(
Z
1
,
A
1
,
Z
2
,
A
2
,
W
1
,
W
2
,
X
,
Y
)
:

 
 
 
 
#
m
 
=
 
Y
.
s
i
z
e

 
 
 
 
o
n
e
_
h
o
t
_
Y
 
=
 
o
n
e
_
h
o
t
(
Y
)

 
 
 
 
d
Z
2
 
=
 
A
2
 
-
 
o
n
e
_
h
o
t
_
Y

 
 
 
 
d
W
2
 
=
 
1
 
/
 
m
 
*
 
d
Z
2
.
d
o
t
(
A
1
.
T
)

 
 
 
 
d
b
2
 
=
 
1
 
/
 
m
 
*
 
n
p
.
s
u
m
(
d
Z
2
,
 
a
x
i
s
 
=
 
1
,
 
k
e
e
p
d
i
m
s
 
=
 
T
r
u
e
)

 
 
 
 
d
Z
1
 
=
 
W
2
.
T
.
d
o
t
(
d
Z
2
)
*
d
e
r
i
v
_
R
e
L
U
(
Z
1
)

 
 
 
 
d
W
1
 
=
 
1
 
/
 
m
 
*
 
d
Z
1
.
d
o
t
(
X
.
T
)

 
 
 
 
d
b
1
 
=
 
1
 
/
 
m
 
*
 
n
p
.
s
u
m
(
d
Z
1
,
 
a
x
i
s
 
=
 
1
,
 
k
e
e
p
d
i
m
s
 
=
 
T
r
u
e
)

 
 
 
 

 
 
 
 
r
e
t
u
r
n
 
d
W
1
,
 
d
b
1
,
 
d
W
2
,
 
d
b
2


d
e
f
 
u
p
d
a
t
e
_
p
a
r
a
m
s
(
W
1
,
 
b
1
,
 
W
2
,
 
b
2
,
 
d
W
1
,
 
d
b
1
,
 
d
W
2
,
 
d
b
2
,
 
a
l
p
h
a
)
:

 
 
 
 
 
 
 
 

 
 
 
 
W
1
 
=
 
W
1
 
-
 
a
l
p
h
a
 
*
 
d
W
1

 
 
 
 
b
1
 
=
 
b
1
 
-
 
a
l
p
h
a
 
*
 
d
b
1

 
 
 
 
W
2
 
=
 
W
2
 
-
 
a
l
p
h
a
 
*
 
d
W
2

 
 
 
 
b
2
 
=
 
b
2
 
-
 
a
l
p
h
a
 
*
 
d
b
2

 
 
 
 
r
e
t
u
r
n
 
W
1
,
 
b
1
,
 
W
2
,
 
b
2




 
 
 
 
```

#
 
P
e
r
f
o
r
m
a
n
c
e



```python
d
e
f
 
g
e
t
_
a
c
c
u
r
a
c
y
(
p
r
e
d
i
c
t
i
o
n
s
,
 
Y
)
:

 
 
 
 
p
r
i
n
t
(
p
r
e
d
i
c
t
i
o
n
s
,
 
Y
)

 
 
 
 
r
e
t
u
r
n
 
n
p
.
s
u
m
(
p
r
e
d
i
c
t
i
o
n
s
 
=
=
 
Y
)
 
/
 
Y
.
s
i
z
e


d
e
f
 
g
e
t
_
p
r
e
d
i
c
t
i
o
n
s
(
A
2
)
:

 
 
 
 
r
e
t
u
r
n
 
n
p
.
a
r
g
m
a
x
(
A
2
,
 
0
)


d
e
f
 
g
r
a
d
i
e
n
t
_
d
e
s
c
e
n
t
(
X
,
 
Y
,
 
a
l
p
h
a
,
i
t
e
r
a
t
i
o
n
s
)
:

 
 
 
 
W
1
,
 
b
1
,
 
W
2
,
 
b
2
 
=
 
i
n
i
t
_
p
a
r
a
m
s
(
)

 
 
 
 
f
o
r
 
i
 
i
n
 
r
a
n
g
e
(
i
t
e
r
a
t
i
o
n
s
)
:

 
 
 
 
 
 
 
 
Z
1
,
 
A
1
,
 
Z
2
,
 
A
2
 
=
 
f
o
r
w
a
r
d
_
p
r
o
p
(
W
1
,
 
b
1
,
 
W
2
,
 
b
2
,
 
X
)

 
 
 
 
 
 
 
 
d
W
1
,
 
d
b
1
,
 
d
W
2
,
 
d
b
2
 
=
 
b
a
c
k
_
p
r
o
p
(
Z
1
,
 
A
1
,
 
Z
2
,
 
A
2
,
 
W
1
,
 
W
2
,
 
X
,
 
Y
)

 
 
 
 
 
 
 
 
W
1
,
 
b
1
,
 
W
2
,
 
b
2
 
=
 
u
p
d
a
t
e
_
p
a
r
a
m
s
(
W
1
,
 
b
1
,
 
W
2
,
 
b
2
,
 
d
W
1
,
 
d
b
1
,
 
d
W
2
,
 
d
b
2
,
 
a
l
p
h
a
)

 
 
 
 
 
 
 
 
i
f
 
i
 
%
 
2
0
 
=
=
 
0
:

 
 
 
 
 
 
 
 
 
 
 
 
p
r
i
n
t
(
'
I
t
e
r
a
t
i
o
n
s
 
:
 
'
,
 
i
)

 
 
 
 
 
 
 
 
 
 
 
 
p
r
i
n
t
(
'
A
c
c
u
r
a
c
y
 
:
 
'
,
 
g
e
t
_
a
c
c
u
r
a
c
y
(
g
e
t
_
p
r
e
d
i
c
t
i
o
n
s
(
A
2
)
,
Y
)
)

 
 
 
 
r
e
t
u
r
n
 
W
1
,
 
b
1
,
 
W
2
,
 
b
2
```

#
 
T
r
a
i
n
 
&
 
T
e
s
t



```python
W
1
,
 
b
1
,
 
W
2
,
 
b
2
 
=
 
g
r
a
d
i
e
n
t
_
d
e
s
c
e
n
t
(
X
_
t
r
a
i
n
,
 
Y
_
t
r
a
i
n
,
 
0
.
1
5
,
 
5
0
0
)
```

<pre>
I
t
e
r
a
t
i
o
n
s
 
:
 
 
0

[
0
 
0
 
0
 
.
.
.
 
0
 
0
 
0
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
0
7
1
0
7
1
4
2
8
5
7
1
4
2
8
5
8

I
t
e
r
a
t
i
o
n
s
 
:
 
 
2
0

[
5
 
0
 
2
 
.
.
.
 
2
 
9
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
3
4
9

I
t
e
r
a
t
i
o
n
s
 
:
 
 
4
0

[
1
 
8
 
2
 
.
.
.
 
2
 
4
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
4
8
9
3
3
3
3
3
3
3
3
3
3
3
3
3
4

I
t
e
r
a
t
i
o
n
s
 
:
 
 
6
0

[
1
 
8
 
2
 
.
.
.
 
2
 
4
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
5
7
3
1
1
9
0
4
7
6
1
9
0
4
7
6

I
t
e
r
a
t
i
o
n
s
 
:
 
 
8
0

[
1
 
8
 
2
 
.
.
.
 
6
 
4
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
6
3
6
5

I
t
e
r
a
t
i
o
n
s
 
:
 
 
1
0
0

[
1
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
6
7
9
4
2
8
5
7
1
4
2
8
5
7
1
4

I
t
e
r
a
t
i
o
n
s
 
:
 
 
1
2
0

[
1
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
7
1
4
2
8
5
7
1
4
2
8
5
7
1
4
3

I
t
e
r
a
t
i
o
n
s
 
:
 
 
1
4
0

[
1
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
7
4
4
0
4
7
6
1
9
0
4
7
6
1
9
1

I
t
e
r
a
t
i
o
n
s
 
:
 
 
1
6
0

[
8
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
7
6
7
7
1
4
2
8
5
7
1
4
2
8
5
7

I
t
e
r
a
t
i
o
n
s
 
:
 
 
1
8
0

[
8
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
7
8
5
3
5
7
1
4
2
8
5
7
1
4
2
9

I
t
e
r
a
t
i
o
n
s
 
:
 
 
2
0
0

[
8
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
7
9
9
9
0
4
7
6
1
9
0
4
7
6
1
9

I
t
e
r
a
t
i
o
n
s
 
:
 
 
2
2
0

[
7
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
1
1
0
7
1
4
2
8
5
7
1
4
2
8
6

I
t
e
r
a
t
i
o
n
s
 
:
 
 
2
4
0

[
7
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
2
1
6
1
9
0
4
7
6
1
9
0
4
7
6

I
t
e
r
a
t
i
o
n
s
 
:
 
 
2
6
0

[
7
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
2
9
4
7
6
1
9
0
4
7
6
1
9
0
5

I
t
e
r
a
t
i
o
n
s
 
:
 
 
2
8
0

[
7
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
3
6
0
4
7
6
1
9
0
4
7
6
1
9

I
t
e
r
a
t
i
o
n
s
 
:
 
 
3
0
0

[
7
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
4
1
9
5
2
3
8
0
9
5
2
3
8
1

I
t
e
r
a
t
i
o
n
s
 
:
 
 
3
2
0

[
7
 
8
 
2
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
4
6
5
4
7
6
1
9
0
4
7
6
1
9

I
t
e
r
a
t
i
o
n
s
 
:
 
 
3
4
0

[
7
 
8
 
4
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
5
0
7
3
8
0
9
5
2
3
8
0
9
5
3

I
t
e
r
a
t
i
o
n
s
 
:
 
 
3
6
0

[
7
 
8
 
4
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
5
4
6
6
6
6
6
6
6
6
6
6
6
6
7

I
t
e
r
a
t
i
o
n
s
 
:
 
 
3
8
0

[
7
 
8
 
4
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
5
8
0
9
5
2
3
8
0
9
5
2
3
8
1

I
t
e
r
a
t
i
o
n
s
 
:
 
 
4
0
0

[
7
 
8
 
4
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
6
1
0
4
7
6
1
9
0
4
7
6
1
9
1

I
t
e
r
a
t
i
o
n
s
 
:
 
 
4
2
0

[
7
 
8
 
4
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
6
4
1
4
2
8
5
7
1
4
2
8
5
7
1

I
t
e
r
a
t
i
o
n
s
 
:
 
 
4
4
0

[
7
 
8
 
4
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
6
6
4
7
6
1
9
0
4
7
6
1
9
0
5

I
t
e
r
a
t
i
o
n
s
 
:
 
 
4
6
0

[
7
 
8
 
4
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
6
8
5
7
1
4
2
8
5
7
1
4
2
8
5

I
t
e
r
a
t
i
o
n
s
 
:
 
 
4
8
0

[
7
 
8
 
4
 
.
.
.
 
6
 
2
 
2
]
 
[
7
 
2
 
2
 
.
.
.
 
6
 
2
 
2
]

A
c
c
u
r
a
c
y
 
:
 
 
0
.
8
7
0
7
8
5
7
1
4
2
8
5
7
1
4
3

</pre>

```python
t
e
s
t
_
p
r
e
d
i
c
t
i
o
n
(
3
7
,
 
W
1
,
 
b
1
,
 
W
2
,
 
b
2
)
```

<pre>
P
r
e
d
i
c
t
i
o
n
:
 
 
[
2
]

L
a
b
e
l
:
 
 
2

</pre>
<pre>
<
F
i
g
u
r
e
 
s
i
z
e
 
4
3
2
x
2
8
8
 
w
i
t
h
 
1
 
A
x
e
s
>
</pre>

```python
```
