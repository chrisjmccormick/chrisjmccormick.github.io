---
layout: post
title:  "Getting Started with mlpack"
date:   2017-02-01 7:00:00 -0800
comments: true
image: assets/mlpack.png
tags: mlpack, C++, Linux, knn
---

I’ve recently needed to perform a benchmarking experiment with k-NN in C++, so I found [mlpack](http://www.mlpack.org) as what appears to be a popular and high-performance machine learning library in C++.

I’m not a very strong Linux user (though I’m working on it!), so I actually had a lot of trouble getting up and going with mlpack, despite their documentation. 

In this guide, I’ll cover the steps needed to get up and running, but also offer some explanation of what’s going in each. So if you’re already an expert at working with C++ libraries in Linux, you may find this post pretty boring :).

Downloading pre-compiled mlpack with the package manager
========================================================
I’m currently running Ubuntu 16.04, so some of this may be Ubuntu-specific.

The Ubuntu package manager helps you get mlpack installed as well as any dependencies (and it appears that mlpack has a lot of dependencies on, e.g., other linear algebra libraries). 

<div class="message">
Note that the package manager is different from the “Ubuntu Software Center”. The software center is more like an app-store, and you won’t find mlpack there.
</div>

The name of the package is ‘libmlpack-dev’. This is going to install the mlpack libraries and header files for you--it does not include the source code for mlpack, which you don’t need if you’re just planning to reference the libraries. It also does *not* include any source examples. They provide a couple code examples as *text* [on their website](http://mlpack.org/docs/mlpack-git/doxygen.php?doc=sample.html); to run these you would create your own .cpp file and paste in the code (but you also need to supply your own dataset! 0_o). More on example code later.

I found the package name a little confusing (why isn't it just "mlpack"?), so here are some clarifications on the "lib" and "-dev" parts of the package name:

* Dynamically-linked libraries like mlpack all have ‘lib’ prepended to their package name (like liblapack, libblas, etc.). 
  * The Dynamic Linker in Linux (called "ld") requires dynamically-linked libraries to have the form lib*.so* ([Reference](http://stackoverflow.com/questions/11842729/ldconfig-only-links-files-starting-with-lib)). 
  * ".so" stands for "Shared Object", and it's analogous to DLLs on Windows.
* The “-dev” suffix on the package name is a convention that indicates that this package contains libraries and header files that you can compile against, as opposed to just executable binaries. ([Reference](http://stackoverflow.com/questions/1157192/what-do-the-dev-packages-in-the-linux-package-repositories-actually-contain))

Another thing that confused me--how would you know the name of the package you want to install if all you know is that the project is called “mlpack”?

[This page](http://www.howtogeek.com/229682/how-to-find-out-exact-package-names-for-applications-in-linux/) provides a nice tutorial (with a lot of detail) about how you can find packages and learn about them. Here’s the command that I would have found most helpful, though: `apt-cache search 'mlpack'`. Those single quotes around mlpack are actually wildcards--they allow it to match any package with mlpack anywhere in the name.

{% highlight text %}
chrismcc@ubuntu:~$ apt-cache search 'mlpack'
libmlpack-dev - intuitive, fast, scalable C++ machine learning library (development libs)
libmlpack2 - intuitive, fast, scalable C++ machine learning library (runtime library)
mlpack-bin - intuitive, fast, scalable C++ machine learning library (binaries)
mlpack-doc - intuitive, fast, scalable C++ machine learning library (documentation)
{% endhighlight %}

Here’s what each of those packages provides.

* libmlpack-dev - If you are going to write code that references the mlpack libraries, this is what you need.
* libmlpack2 - If you’re not programming with mlpack, but you’re using an application that uses the mlpack libraries, then you’d just need this package with the “runtime library” (the dynamically-linked library).
* mlpack-bin - The mlpack project actually includes command line tool versions of some of the machine learning algorithms it implements. So, for example, you could run k-means clustering on a dataset from the command line without doing any programming. This package contains those binaries.
* mlpack-doc - Documentation for the libraries.

So to write our own code using the mlpack libraries, we just need libmlpack-dev. Grab it with the APT (Advanced Packaging Tool) package manager with the following command:

{% highlight text %}
sudo apt-get install libmlpack-dev
{% endhighlight %}

This will install mlpack and all of the libraries it depends on. Except one, apparently--you'll also need to install Boost:

{% highlight text %}
sudo apt-get install libboost-all-dev
{% endhighlight %}

Maybe Boost was left out of the dependency list because it's so commonly used? I don't know.

Install location
================
Something that left me pretty confused from the installation was that I had no idea where mlpack was installed to. (Mainly, I wanted to know this because I assumed it would have installed some example source files for me somewhere, but I learned later that it doesn’t include any.)

To list out all of the files installed by mlpack, use this command:

{% highlight text %}
dpkg -L libmlpack-dev
{% endhighlight %}

There are some default locations for libraries in Linux, and that’s where you’ll find mlpack:

* It installs lots of headers under /usr/include/mlpack/. 
* It installs the library file to /usr/lib/x86_64-linux-gnu/libmlpack.so 

These default locations are already part of the path for gcc / g++, so you're all set to #include the mlpack header files in your code!

<div class="message">
What's "/usr/"? Is that like my User directory on Windows? (Answer: No.)
Linux usually starts you out in your ‘home’ directory, e.g. /home/chrismcc/. This is where you find your personal files (documents, desktop, pictures, etc.).  You can also refer to your home directory by just tilde ‘~’. This used to confuse me because I assumed ~ was the symbol for the root of the file system--it’s not! just ‘/’ is the root directory. /usr/ is a directory under root where installed software lives. 
</div>


Compiling and Running an Example
================================
As a first example, we'll use the [sample code](http://mlpack.org/docs/mlpack-git/doxygen.php?doc=sample.html) from the mlpack site for doing a nearest neighbor search.

This very simple example takes a dataset of vectors and finds the nearest neighbor for each data point. It uses the dataset both as the reference and the query vectors.

I've taken their original example and added some detailed comments to explain what's going on.

Relevant Documentation:

* [NeighborSearch](http://www.mlpack.org/docs/mlpack-2.0.2/doxygen.php?doc=classmlpack_1_1neighbor_1_1NeighborSearch.html)
  * [Constructor](http://www.mlpack.org/docs/mlpack-2.0.2/doxygen.php?doc=classmlpack_1_1neighbor_1_1NeighborSearch.html#a16cb809195a197f551abd71517b4e724)
  * [Search](http://www.mlpack.org/docs/mlpack-2.0.2/doxygen.php?doc=classmlpack_1_1neighbor_1_1NeighborSearch.html#a9fa8cb63a20f46a13eda80009e72fd09)
* [data::Load](http://www.mlpack.org/docs/mlpack-2.0.2/doxygen.php?doc=namespacemlpack_1_1data.html#ae8cd401ac166a40e1e836f752814402b)

Save the following source code in a file called knn_example.cpp:

{% highlight cpp %}
/*
 * ======== knn_example.cpp =========
 * This very simple example takes a dataset of vectors and finds the nearest 
 * neighbor for each data point. It uses the dataset both as the reference and
 * the query vectors.
 *
 * mlpack documentation is here:
 * http://www.mlpack.org/docs/mlpack-2.0.2/doxygen.php
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace mlpack;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance

int main()
{
    // Armadillo is a C++ linear algebra library; mlpack uses its matrix data type.
    arma::mat data;
    
    /*
     * Load the data from a file. mlpack does not provide an example dataset, 
     * so I've included a tiny one.
     *
     * 'data' is a helper class in mlpack that facilitates saving and loading 
     * matrices and models.
     *
     * Pass the filename, matrix to hold the data, and set fatal = true to have
     * it throw an exception if there is an issue.
     *
     * 'Load' excepts comma-separated and tab-separated text files, and will 
     * infer the format.
     */
    data::Load("data.csv", data, true);
    
    /* 
     * Create a NeighborSearch model. The parameters of the model are specified
     * with templates:
     *  - Sorting method: "NearestNeighborSort" - This class sorts by increasing
     *    distance.
     *  - Distance metric: "ManhattanDistance" - The L1 distance, sum of absolute
     *    distances.
     *
     * Pass the reference dataset (the vectors to be searched through) to the
     * constructor.
     */
    NeighborSearch<NearestNeighborSort, ManhattanDistance> nn(data);
    
    /*
     * Create the matrices to hold the results of the search. 
     *   neighbors [k  x  n] - Indeces of the nearest neighbor(s). 
     *                         One column per data query vector and one row per
     *                        'k' neighbors.
     *   distances [k  x  n] - Calculated distance values.
     *                         One column per data query vector and one row per
     *                        'k' neighbors.
     */
    arma::Mat<size_t> neighbors;
    arma::mat distances; 
    
    /*
     * Find the nearest neighbors. 
     *
     * If no query vectors are provided (as is the case here), then the 
     * reference vectors are searched against themselves.
     *
     * Specify the number of neighbors to find, k = 1, and provide matrices
     * to hold the result indeces and distances.
     */ 
    nn.Search(1, neighbors, distances);
    
    // Print out each neighbor and its distance.
    for (size_t i = 0; i < neighbors.n_elem; ++i)
    {
        std::cout << "Nearest neighbor of point " << i << " is point "
        << neighbors[i] << " and the distance is " << distances[i] << ".\n";
    }
}
{% endhighlight %}

And save this toy dataset as data.csv:

{% highlight text %}
3,3,3,3,0
3,4,4,3,0
3,4,4,3,0
3,3,4,3,0
3,6,4,3,0
2,4,4,3,0
2,4,4,1,0
3,3,3,2,0
3,4,4,2,0
3,4,4,2,0
3,3,4,2,0
3,6,4,2,0
2,4,4,2,0
{% endhighlight %}

To compile the example, you'll use g++ (the C++ equivalent of gcc).

{% highlight text %}
$ g++ knn_example.cpp -o knn_example -std=c++11 -larmadillo -lmlpack -lboost_serialization
{% endhighlight %}

* knn_example.cpp - The code to compile.
* -o knn_example - The binary (executable) to output.
* -std=c++11 - mlpack documentation says you need to set this flag.
* -larmadillo -lmlpack -lboost_serialization - The "-l" flag tells the linker to look for these libraries.
  * armadillo is a linear algebra library that mlpack uses.

Finally, to run the example, execute the binary:

{% highlight text %}
$ ./knn_example
{% endhighlight %}

<div class="message">
Don't leave out the "./"! In Windows, you can just type the name of an executable in the current directory and hit enter and it will run. In Linux, if you want to do the same you need to prepend the "./".
</div>

And you should see the following output:

```
Nearest neighbor of point 0 is point 7 and the distance is 1.
Nearest neighbor of point 1 is point 2 and the distance is 0.
Nearest neighbor of point 2 is point 1 and the distance is 0.
Nearest neighbor of point 3 is point 10 and the distance is 1.
Nearest neighbor of point 4 is point 11 and the distance is 1.
Nearest neighbor of point 5 is point 12 and the distance is 1.
Nearest neighbor of point 6 is point 12 and the distance is 1.
Nearest neighbor of point 7 is point 10 and the distance is 1.
Nearest neighbor of point 8 is point 9 and the distance is 0.
Nearest neighbor of point 9 is point 8 and the distance is 0.
Nearest neighbor of point 10 is point 9 and the distance is 1.
Nearest neighbor of point 11 is point 4 and the distance is 1.
Nearest neighbor of point 12 is point 9 and the distance is 1.
```

You're up and running!

