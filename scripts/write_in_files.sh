# /usr/bin/bash
# ----------------------------------#
# write_in_files.sh                 #
# ----------------------------------#
# Reads the lines in the
# run_parameters.dat file (this
# is created by the generate_
# production_run_table.py script) and
# takes a template disc.in file and
# creates one for each of the runs
# entering the values for EOS params
# ----------------------------------#
# Author: Adam Fenton
# Date:   20220728
# ----------------------------------#
counter=0
while read -r line; do
    counter=$((counter+1))
    run_dir="run_"`printf %03d $counter`
    cp disc_template.in $run_dir/disc.in
    cp disc_setup_template.setup $run_dir/disc.setup
    stringarray=($line)
    new_rc1=${stringarray[0]}
    new_rc2=${stringarray[1]}
    new_rc3=${stringarray[2]}

    new_g1=${stringarray[3]}
    new_g2=${stringarray[4]}
    new_g3=${stringarray[5]}

    new_HonR=${stringarray[7]}
    echo $new_HonR


    rhocrit1=`grep -w rhocrit1 $run_dir/disc.in | awk '{print $3}'`;sed -i '' "s|$rhocrit1|$new_rc1|g" $run_dir/disc.in
    rhocrit2=`grep -w rhocrit2 $run_dir/disc.in | awk '{print $3}'`;sed -i '' "s|$rhocrit2|$new_rc2|g" $run_dir/disc.in;
    rhocrit3=`grep -w rhocrit3 $run_dir/disc.in | awk '{print $3}'`;sed -i '' "s|$rhocrit3|$new_rc3|g" $run_dir/disc.in;

    gamma1=`grep -w gamma1 $run_dir/disc.in | awk '{print $3}'`;sed -i '' "s|$gamma1|$new_g1|g" $run_dir/disc.in;
    gamma2=`grep -w gamma2 $run_dir/disc.in | awk '{print $3}'`;sed -i '' "s|$gamma2|$new_g2|g" $run_dir/disc.in;
    gamma3=`grep -w gamma3 $run_dir/disc.in | awk '{print $3}'`;sed -i '' "s|$gamma3|$new_g3|g" $run_dir/disc.in;

    HonR=`grep -w  H_R $run_dir/disc.setup | awk '{print $3}'`;sed -i '' "s|$HonR|$new_HonR|g" $run_dir/disc.setup;

done < run_parameters.dat
