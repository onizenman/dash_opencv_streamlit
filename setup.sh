mkdir -p ~/.streamlit/
echo "
[server]\n
headless = true\n
enableCORS=false\n
enableXsrfProtection=false\n
port = $PORT\n
[browser]\n
gatherUsageStats = false\n
\n
" > ~/.streamlit/config.toml
