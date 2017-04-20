ls#!/bin/bash 
#Note: Preprocessing for annot.opcorpora.xml(http://opencorpora.org)
#Needed
#preprocess dataset_file_name

deleting_tag1='<source>'
deleting_tag2='</source>'
final_file='cyrillic_annot.opcorpora.txt'

if [ -z "$1" ]
then
    echo "For begin type 'preprocess dataset_file_name'"
else
    temp_file1=$(mktemp tmp.XXXX)
    temp_file2=$(mktemp tmp.XXXX)
    temp_file3=$(mktemp tmp.XXXX)
    temp_file4=$(mktemp tmp.XXXX)

    cat $1 |grep source >> $temp_file1

    echo '' > $final_file

    while read line
    do
        line=${line/$deleting_tag1/''}
        line=${line/$deleting_tag2/''}
        echo $line >> $temp_file2

    done < $temp_file1
    
    cat $temp_file2 | grep -v --null [$'\x41'-$'\x5A'$'\x61'-$'\x7A'] >> $temp_file3
    templete1='[\p{Arabic}\p{Armenian}\p{Bengali}\p{Bopomofo}\p{Braille}\p{Buhid}\p{Canadian_Aboriginal}\p{Cherokee}\p{Devanagari}\p{Ethiopic}\p{Georgian}\p{Greek}\p{Gujarati}\p{Gurmukhi}\p{Han}\p{Hangul}\p{Hanunoo}\p{Hebrew}\p{Hiragana}\p{Inherited}\p{Kannada}\p{Katakana}\p{Khmer}\p{Lao}\p{Latin}\p{Limbu}\p{Malayalam}\p{Mongolian}\p{Myanmar}\p{Ogham}\p{Oriya}\p{Runic}\p{Sinhala}\p{Syriac}\p{Tagalog}\p{Tagbanwa}\p{Tamil}\p{Telugu}\p{Thaana}\p{Thai}\p{Tibetan}\p{Yi}]'
    cat $temp_file3 |  grep -v -P $templete1 >> $temp_file4
    templete2=[→‰●½‑ѲѢ§≥″ў₤™¤∞†ї≤˚☿‚·©↔$'\uf04a'$'\uf04c'$'\uf0b8'$'\xa0'$'\ufeff'$'\xad'$'{'$'#'$'|'$'}'$'^']
    cat $temp_file4 |  grep -v --null $templete2 >> $final_file

    rm tmp.*
fi

#\p{Arabic}
#\p{Armenian}
#\p{Bengali}
#\p{Bopomofo}
#\p{Braille}
#\p{Buhid}
#\p{Canadian_Aboriginal}
#\p{Cherokee}
#\p{Devanagari}
#\p{Ethiopic}
#\p{Georgian}
#\p{Greek}
#\p{Gujarati}
#\p{Gurmukhi}
#\p{Han}
#\p{Hangul}
#\p{Hanunoo}
#\p{Hebrew}
#\p{Hiragana}
#\p{Inherited}
#\p{Kannada}
#\p{Katakana}
#\p{Khmer}
#\p{Lao}
#\p{Latin}
#\p{Limbu}
#\p{Malayalam}
#\p{Mongolian}
#\p{Myanmar}
#\p{Ogham}
#\p{Oriya}
#\p{Runic}
#\p{Sinhala}
#\p{Syriac}
#\p{Tagalog}
#\p{Tagbanwa}
#\p{TaiLe}
#\p{Tamil}
#\p{Telugu}
#\p{Thaana}
#\p{Thai}
#\p{Tibetan}
#\p{Yi}
