cd /nethome/frugt001/galaxy_b_modes
source /venv/bin/activate
python3 -c "import hdfstream; r=hdfstream.open('cosma','/'); print('network OK', r['FLAMINGO'])"