http://ftp.mozilla.org/pub/mozilla.org/firefox/releases/24.0b7/linux-x86_64/en-US/firefox-24.0b7.tar.bz2

Lower values indicate that the kernel should wait longer to swap. The default value is 60.
1

>> sysctl -w vm.swappiness=10

Or set vm.swappiness = 10 in your /etc/sysctl.conf file.