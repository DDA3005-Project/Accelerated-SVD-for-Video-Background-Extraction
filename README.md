# Project
Hi! This is the project for DDA3005, CUHKSZ. We conduct a recursion accelerated SVD algorithm and leverage it in video background extraction.

Update the naive code for prob1_1, this py can change any matrix to bidiagonal form.
Update the naive code for prob1_3, this file can calculate the SVD for the square matrix after bidiagonalization. 

Prob1_3 still needs to improve the performance of the accelerrated Cholesky, by avoiding using append, diagonal, etc. Try to finish the algorithm by only changing values instead of creating a new list. 
\-- The accelerated Cholesky is tested on recursion decomposition.

Other .ipynb that involved video background construction depends on the basic algorithms mentioned above.
