---
layout: page
permalink: /talks_n/
title: invited talks v2
description: talks that I have given or am yet to give
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="talks">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% talks -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
