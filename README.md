Feature Transformation Algorithms
======
I manually implemented some feature transformation algorithms with Python3. Currently, this project supports the algorithms as follow:

1. Principal Component Analysis (PCA)
2. Fisher Discriminant Analysis (FDA)
3. Kernel-Principal Component Analysis (KernelPCA)
4. Kernel-Fisher Discriminant Analysis (Kernel FDA)

I designed the algorithms in Scikit-learn library style, which means each class implementing the algorithms has `fit`, `transform`, and `fit_transform` methods.
These methods work exactly same with the methods in Scikit-learn library. 
Even if I referred to the library, all algorithms perfectly work without the library.
One important things in my project is that the project supports the Kernel-FDA, which is unsupported by Scikit-learn library.
Feel free to use my implementation!

##Usage
For PCA,
```
>> from PCA import PCA
>> pca = PCA(n_components=2)
>> X_ = pca.fit_transform(X)
```
For FDA,
```
>> from FDA import FisherDiscriminantAnalysis
>> fda = FisherDiscriminantAnalysis(n_components=2)
>> X_ = fda.fit_transform(X)
```
For, Kernel-PCA
```
>> from kernelPCA import KernelPrincipalComponentsAnalysis
>> kpca = KernelPrincipalComponentsAnalysis(n_components=2, kernel='rbf', gamma=2)
>> X_ = kpca.fit_transform(X)
```
For, Kernel-FDA
```
>> from kernelFDA import KernelFisherDiscriminantAnalysis
>> kfda = KernelFisherDiscriminantAnalysis(n_components=2, kernel='rbf', gamma=2)
>> X_ = kfda.fit_transform(X)
```
Note that, in current implementation, Kernel methods only support `rbf` and `linear` kernel.
However, it is easy to extend the kernel function, such as `tanh` or `poly`.
Obviously, if you set `linear` as kernel function in both kernel methods, 
the results are exactly same with the standard methods.