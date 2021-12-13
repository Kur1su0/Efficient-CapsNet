echo "####effi####"
python efficient_capsnet.py | tee efficient_cap.log
cat config.json >> efficient_cap.log
