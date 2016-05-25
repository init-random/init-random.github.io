---
layout: post
title:  "Linux Notes"
date: 2016-02-17 23:28:57 -0500
categories: regression
---

Linux 64-bit firefox build may be found here.

http://ftp.mozilla.org/pub/firefox/releases/47.0b8/linux-x86_64/en-US/firefox-47.0b8.tar.bz2

For setting the propensity to swap you can set the swappiness parameter. Lower values indicate that the kernel 
should wait longer to swap. The default value is 60.

{% highlight bash %}
>> sysctl -w vm.swappiness=10
{% endhighlight %}

Or for a persistent configuration set vm.swappiness = 10 in your /etc/sysctl.conf.

One of most important productivity boosters is to swap your caplock and control keys. The control key is one
of the most used keys for shortcuts. So, why have this key in remote place on the keyboard? In XFCE, if you go
to the control panel click on the Application Autostart tab and click Add. In the command section set the following
 
{% highlight bash %}
/usr/bin/setxkbmap -option "ctrl:swapcaps"
{% endhighlight %}

with the other fields set to something like ctrlswap. Upon next login your your caplock will be the control key
 and the control key the caplock.
