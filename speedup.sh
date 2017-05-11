(./runloop.sh python main.py -d data/polya.csv) > speedup1.txt
(./runloop.sh python main.py -d data/polya.csv -p -n 2) > speedup2.txt
(./runloop.sh python main.py -d data/polya.csv -p -n 4) > speedup4.txt
(./runloop.sh python main.py -d data/polya.csv -p -n 8) > speedup8.txt
(./runloop.sh python main.py -d data/polya.csv -p -n 16) > speedup16.txt
