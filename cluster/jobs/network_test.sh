cd /nethome/frugt001/galaxy_b_modes
conda activate b-modes
python3 -c "import hdfstream; r=hdfstream.open('cosma','/'); print('network OK', r['FLAMINGO'])"
