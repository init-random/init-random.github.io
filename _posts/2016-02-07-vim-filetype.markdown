---
layout: post
comments: true
title:  "VIM Filetype"
date: 2016-02-07 23:28:57 -0500
categories: vim
---

There are many times when I am working on a project and I do some
“throw away” editing in Vim. For example, I might write some SQL
queries in Vim and copy them into my client. I wanted a way to improve
my SQL editing in Vim. This function creates a temporary file of the
specified file type. This is for UNIX-based systems.
	

{% highlight bash %}
function AsTmpFile(...)
  if !exists("~/.vim/tmp")
      let l:t=system("mkdir ~/.vim/tmp")
      unlet l:t
  endif
  let l:fname="_".system("date | md5sum | cut -d' ' -f1")
  let l:fname = substitute(l:fname, '\n$', '', '')
  let l:suffix=".txt"
  if exists("a:1")
      let l:suffix=".".a:1
  endif
  let l:fname=l:fname . l:suffix
  execute ":w ~/.vim/tmp/" . l:fname
endfunction
{% endhighlight %}

There default file extension will be .txt. A different extension may be added as an argument. Files will be
written to ~/.vim/tmp/; create this directory if it does not exist. This file will be retained for later use. 

So, now I might run the following ex commands.

{% highlight bash %}
:tabnew
:call AsTmpFile("sql")
{% endhighlight %}

This will create a file in the said directory with a .sql suffix. The benefit of this is that we can take advantage of Vim filetypes. So, I might have the following in my .vimrc to enable filetypes.
	
{% highlight bash %}
filetype on
filetype indent on
filetype plugin on
{% endhighlight %}

Then I create a ~/.vim/ftplugin/sql.vim file which tells vim that if it sees a *.sql file, then apply the settings specified in sql.vim. The contents of this file is the following:

{% highlight bash %}
set dictionary=~/.vim/dict/mysql-keywords.txt,~/.vim/dict/db-schema.txt
set complete=.,k
{% endhighlight %}

The first line sets two dictionaries for sql files, see [here][vimdict]. The first is a list of MySQL keywords and the second is the 
database(s) schema. This should be enough to auto-complete about 90% of your SQL code. The second property enables 
dictionary completion to ctrl-p/ctrl-n. Default dictionary completion is ctrl-x ctrl-k which is too cumbersome. The 
second line also only looks for completions in the current buffer and the dictionary. So you should be able to now type
	
{% highlight bash %}
SEL<ctrl-n> # compelte SELECT from keywords file
SELECT pri<ctrl-n> # complete price from schema file
{% endhighlight %}

{% include comments.md page-identifier="vim_filetype" %} 

[vimdict]: https://github.com/init-random/dotfiles/tree/master/.vim/dict