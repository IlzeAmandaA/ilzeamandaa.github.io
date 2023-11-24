---
layout: page
permalink: /talks/
title: invited talks
description: talks that I have given or am yet to give
years: [2023]
nav: true
nav_order: 1
---

<div class="talks">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
    {% bibliography -f talks -q @*[year={{y}}]* %}
{% endfor %}

</div>