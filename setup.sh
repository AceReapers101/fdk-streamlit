python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.7/index.html
mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"cheah3838@gmail.com\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml