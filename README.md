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

Time Zone
=========
Initially, I didn't have a timezone set for the site. This actually caused a problem at one point, because the site builder generated my post with a different date than what I put in the post, after converting the time. This was fixed by setting the `timezone: America/Los_Angeles` in `_config.yml`.

Math & Equations
================
Adding support for equations with MathJax was a cinch. I just put the following line in `_includes/head.html`

<code>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
</code>

{% highlight html %}
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
{% endhighlight %}

[poole_repo]: https://github.com/poole/poole "Poole repository on GitHub"
[jl_poole]: http://joshualande.com/jekyll-github-pages-poole/ "Joshua Lande's blog post on Jekyll with Poole"
