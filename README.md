My blog, dedicated to Machine Learning tutorials, code, and insights.

Base Design
===========
My blog is built on top of [Poole][poole_repo]. I took inspiration and a lot of direction from Joshua Lande and his blog post [here][jl_poole].

Because my blog is my GitHub user page (and not just a project page), I worked from the master branch of Poole.

Page Links in Header
====================
Following instructions in [Joshua Lande's post][jl_poole], I modified the masthead of the 'default.html' layout and added his code for displaying links to the top level pages. Also, the list of top level pages is defined in _config.yml.

Google Analytics
================
Google Analytics is enabled by simply setting the variable `analytics_id` in `_config.yml`.

[poole_repo]: https://github.com/poole/poole "Poole repository on GitHub"
[jl_poole]: http://joshualande.com/jekyll-github-pages-poole/ "Joshua Lande's blog post on Jekyll with Poole"
