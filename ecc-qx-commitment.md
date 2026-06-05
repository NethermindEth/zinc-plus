# ECC Commitment for a ℚ[X]-Valued Oracle

So you can have a  ℚ[X]-valued polynomial represented as the following

```
f_b(X) = Σ_{j<d} c_{b,j} X^j,    c_{b,j} ∈ ℚ
```

We can flatten it to make one giant c⃗ vector. 

```
c⃗ = (c_{b,j})_{b∈{0,1}^ν, j<d}
```

Then commit:

```text
C_f = Σ_{b,j} [c_{b,j}]ᵣ · G_{b,j} + ρH
```
