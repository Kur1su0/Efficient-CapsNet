echo "####effi####"
python efficient_capsnet.py | tee efficient_cap.log
cat config.json >> efficient_cap.log

echo "####ori####"
python original_capsnet.py | tee ori_cap.log
cat config.json >> ori_cap.log