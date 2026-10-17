"""Local Streamlit workspace for vector search, data inspection, and index tuning."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.ui.client import request, API_URL

st.set_page_config(page_title='RAG Indexing Lab', page_icon='🔎', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {border:1px solid #dfe6ef; border-radius:12px; padding:14px;}
[data-testid="stSidebar"] {border-right:1px solid #dfe6ef;}
</style>''', unsafe_allow_html=True)


def api(method, path, payload=None, params=None):
    try:
        return request(method, path, payload, params)
    except Exception as error:
        st.error(f'Local API request failed: {error}')
        st.caption('Start the application with: python scripts/run_ui.py. Keep Docker Desktop running for PostgreSQL and Qdrant.')
        return None


@st.cache_data(ttl=5, show_spinner=False)
def overview_data():
    return request('GET', '/overview')



@st.cache_data(ttl=60, show_spinner=False)
def labeled_queries(split):
    return request('GET', '/evaluation/queries', params={'split': split})


def query_controls(prefix, label):
    source = st.radio('Query source', ['Free text', 'SciFact labeled query'], horizontal=True, key=prefix + '_source')
    if source == 'Free text':
        return st.text_area(label, value='cancer immunotherapy', height=90, key=prefix + '_query'), None, 'test'
    split = st.selectbox('Relevance label split', ['test', 'train'], key=prefix + '_split')
    try:
        catalog = labeled_queries(split)
    except Exception as error:
        st.error(f'Cannot load relevance labels: {error}')
        return '', None, split
    if not catalog['available'] or not catalog['queries']:
        st.warning(catalog.get('reason', 'No labeled queries have relevant documents in this sample.'))
        return '', None, split
    entries = {entry['query_id']: entry for entry in catalog['queries']}
    selected = st.selectbox('Labeled query', list(entries), key=prefix + '_query_id',
                            format_func=lambda key: f"{key} · {entries[key]['text']}")
    entry = entries[selected]
    st.info(entry['text'])
    st.caption(f"{entry['relevant_in_sample']} judged relevant document(s) in the 1,000-document sample; {entry['relevant_in_full_corpus']} in the full corpus. Filters can reduce the available count.")
    return entry['text'], selected, split


def metric_value(value):
    return 'N/A' if value is None else f'{value:.1%}'


def search_metrics(metrics):
    if not metrics:
        st.caption('Run the search again to calculate precision and recall.')
        return
    for title, key in [('Labeled relevance', 'relevance'), ('Exact-neighbor agreement', 'exact_neighbors')]:
        values = metrics[key]
        st.caption(title)
        left, right = st.columns(2)
        left.metric(f"Precision@{values['requested_k']}", metric_value(values['precision_at_k']),
                    help='Matching reference documents in the first K results / requested K. Missing result slots count against precision.')
        right.metric(f"Recall@{values['requested_k']}", metric_value(values['recall_at_k']),
                     help='Matching reference documents / reference positives in this search scope. Relevance uses SciFact labels; neighbor agreement uses NumPy top-K.')
        if not values['available']:
            st.caption(values['reason'])
        else:
            st.caption(f"Matched: {values['matched_count']} · Returned: {values['returned_count']} · Recall reference count: {values['recall_denominator']}")
        if key == 'relevance' and 'relevant_excluded_by_sample_or_filters' in values:
            st.caption(f"Scope: searchable sample after filters. Relevant documents excluded by sampling/filters: {values['relevant_excluded_by_sample_or_filters']}.")
    with st.expander('Metric definitions and references'):
        st.write('Precision@K = matched reference documents / requested K. Recall@K = matched reference documents / available reference positives. Empty reference sets are N/A. Labeled relevance treats positive SciFact qrels as relevant and other results as non-relevant for scoring. Exact-neighbor agreement measures index accuracy; it does not demonstrate relevance.')
        st.json(metrics)


def filters(prefix):
    left, middle, right = st.columns(3)
    groups = left.multiselect('Groups', ['alpha', 'beta', 'gamma', 'delta'], key=prefix + '_groups')
    statuses = middle.multiselect('Review status', ['pending', 'reviewed', 'approved'], key=prefix + '_status')
    use_years = right.checkbox('Filter by test year', key=prefix + '_use_years')
    years = right.slider('Test year range', 2022, 2025, (2022, 2025), key=prefix + '_years')
    return {'groups': groups, 'review_statuses': statuses,
            'year_min': years[0] if use_years else None, 'year_max': years[1] if use_years else None}



def hits_view(result):
    st.metric('Search call', f"{result['latency_ms']:.2f} ms")
    st.caption(result['index_type'])
    search_metrics(result.get('metrics', {}))
    hits = result['hits']
    if not hits:
        st.info('No matching documents.')
        return
    rows = [{'Rank': i, 'ID': hit['document']['id'], 'Cosine': round(hit['score'], 6), 'Title': hit['document']['title'],
             'Group': hit['document']['metadata'].get('lab_group'), 'Year': hit['document']['metadata'].get('lab_year')}
            for i, hit in enumerate(hits, 1)]
    st.dataframe(rows, hide_index=True, width='stretch')
    selected = st.selectbox('Inspect result', range(len(hits)), format_func=lambda i: f"{i + 1}. {hits[i]['document']['title'][:80]}",
                            key='hit_' + result['backend'])
    hit = hits[selected]
    st.write(hit['document']['abstract'])
    with st.expander('Document metadata'):
        st.json(hit['document']['metadata'])


st.sidebar.title('RAG Indexing Lab')
st.sidebar.caption('A local vector-search workspace')
st.sidebar.link_button('View search flow diagram', API_URL + '/search-flow')
page = st.sidebar.radio('Workspace', ['Overview', 'Search playground', 'Data explorer', 'Index tuning'])
if st.sidebar.button('Refresh data and backend status'):
    st.cache_data.clear()
try:
    overview = overview_data()
except Exception as error:
    st.error(f'The local API is unavailable: {error}')
    st.code('python scripts/run_ui.py', language='bash')
    st.stop()
st.sidebar.caption(f"{overview['document_count']:,} documents · {overview['dimensions']} dimensions")
st.sidebar.caption('SciFact abstracts · local CPU embeddings')
st.title(page)

if page == 'Overview':
    st.write('Inspect the same saved embeddings across FAISS, PostgreSQL with pgvector, and Qdrant.')
    columns = st.columns(4)
    for column, label, value in zip(columns, ['Documents', 'Vector dimensions', 'Model token limit', 'Truncated inputs'],
                                    [overview['document_count'], overview['dimensions'], overview['model_token_limit'], overview['truncated_documents']]):
        column.metric(label, value)
    st.caption('Truncation affects the embedding input. Complete titles and abstracts remain available in the data explorer.')
    for column, status in zip(st.columns(3), overview['backends']):
        with column:
            st.subheader(status['backend'].upper())
            if status['ready']:
                st.success(f"Ready · {status['stats']['count']:,} vectors")
                st.write(status['stats']['index_type'])
                with st.expander('Inspect backend details'):
                    st.json(status['details'])
            else:
                st.error(status.get('error', 'Dataset is not fully ingested'))
    st.subheader('Metadata available for filter experiments')
    for column, (field, counts) in zip(st.columns(3), overview['metadata_distributions'].items()):
        with column:
            st.caption(field)
            st.bar_chart(pd.DataFrame({'Documents': counts}))
    st.info('Group, year, and review-status labels are synthetic test data, unrelated to actual publication metadata.')
    with st.expander('How the current stores search'):
        st.write('FAISS Flat compares all eligible vectors. PostgreSQL performs exact cosine scans. Qdrant requests exact cosine search with HNSW disabled. Metadata filtering is a separate operation from choosing the nearest vectors.')
        st.write('Index tuning builds temporary HNSW experiments. Existing baseline indexes and source embeddings are preserved.')

elif page == 'Search playground':
    st.write('Embed a query once, apply the same filters, and compare exact retrieval side by side.')
    query, query_id, qrels_split = query_controls('search', 'Search query')
    with st.form('search_form'):
        left, right = st.columns(2)
        backends = left.multiselect('Backends', ['faiss', 'pgvector', 'qdrant'], default=['faiss', 'pgvector', 'qdrant'])
        top_k = right.slider('Top K', 1, 50, overview['defaults']['top_k'])
        chosen_filters = filters('search')
        submitted = st.form_submit_button('Search all selected backends', type='primary')
    if submitted:
        if not query.strip() or not backends:
            st.warning('Enter a query and select at least one backend.')
        else:
            with st.spinner('Embedding the query and searching local stores…'):
                data = api('POST', '/search', {'query': query, 'top_k': top_k, 'backends': backends, 'filters': chosen_filters, 'query_id': query_id, 'qrels_split': qrels_split})
            if data is not None:
                st.session_state['last_search'] = data
    data = st.session_state.get('last_search')
    if data:
        st.subheader(f"Results for: {data['query']}")
        st.caption(f"Eligible documents: {data['eligible_documents']:,} · Query embedding: {data['query_embedding_ms']:.2f} ms · Query tokens: {data['query_tokens']}")
        evaluation = data.get('evaluation', {})
        if evaluation and not evaluation.get('available'):
            st.info(evaluation['reason'])
        if data['query_truncated']:
            st.warning('The query exceeded the embedding model token limit and was truncated.')
        with st.expander('Applied filters'):
            st.json(data['filters'])
        for error in data['errors']:
            st.error(f"{error['backend']}: {error['error']}")
        if data['results']:
            for column, result in zip(st.columns(len(data['results'])), data['results']):
                with column:
                    st.subheader(result['backend'].upper())
                    st.caption('Exact result checked against the filtered NumPy baseline ✓')
                    hits_view(result)
            figure = px.bar([{'Backend': result['backend'], 'Search call (ms)': result['latency_ms']} for result in data['results']],
                            x='Backend', y='Search call (ms)', color='Backend')
            st.plotly_chart(figure, width='stretch')
            st.caption('Single-call observations include client transport and decoding. They exclude model loading and query embedding, and are not P50/P95 benchmark statistics.')
        st.download_button('Download search results', json.dumps(data, indent=2), 'search-results.json', 'application/json')

elif page == 'Data explorer':
    st.write('Browse prepared documents with their saved metadata updates, then inspect a record and vector from a selected backend.')
    with st.form('browse_form'):
        text = st.text_input('Find a title or document ID')
        selected_filters = filters('browse')
        if st.form_submit_button('Apply document filters'):
            st.session_state['browse_request'] = {'text': text, **selected_filters}
    left, right = st.columns(2)
    limit = left.selectbox('Rows per page', [25, 50, 100], index=1)
    page_number = right.number_input('Page', min_value=1, value=1, step=1)
    params = {key: value for key, value in st.session_state.get('browse_request', {}).items() if value is not None}
    params.update({'limit': limit, 'offset': (page_number - 1) * limit})
    data = api('GET', '/documents', params=params)
    if data:
        st.caption(f"{data['total']:,} matching documents · page {page_number} · {data['source']}")
        st.dataframe(data['rows'], hide_index=True, width='stretch')
        if data['rows']:
            title_map = {row['id']: row['title'] for row in data['rows']}
            left, right = st.columns(2)
            stable_id = left.selectbox('Document to inspect', list(title_map), format_func=lambda key: f'{key} · {title_map[key][:80]}')
            backend = right.selectbox('Read document and vector from', ['prepared', 'faiss', 'pgvector', 'qdrant'])
            record = api('GET', f'/documents/{stable_id}', params={'backend': backend})
            if record:
                left, right = st.columns([2, 1])
                with left:
                    st.subheader(record['document']['title'])
                    st.write(record['document']['abstract'])
                with right:
                    st.metric('Vector dimensions', record['dimensions'])
                    st.caption(f"L2 norm: {record['vector_norm']:.8f}")
                    st.caption(f"Input tokens: {record['token_info']['tokens']} · Truncated: {record['token_info']['truncated']}")
                    st.json(record['document']['metadata'])
                st.subheader('Stored vector')
                figure = go.Figure(go.Scatter(x=list(range(record['dimensions'])), y=record['vector'], mode='lines'))
                figure.update_layout(xaxis_title='Dimension', yaxis_title='Value', height=280, margin=dict(t=10))
                st.plotly_chart(figure, width='stretch')
                st.caption('Dimensions are learned features; individual coordinates do not correspond to named words or concepts.')
                with st.expander('All vector values'):
                    st.json(record['vector'])
                st.download_button('Download record and vector', json.dumps(record, indent=2), f'document-{stable_id}.json', 'application/json')
        else:
            st.info('No rows on this page. Try page 1 or adjust your filters.')

elif page == 'Index tuning':
    st.write('Build a temporary HNSW index, measure search, and compare its neighbors with exact NumPy ground truth.')
    st.caption(f"Experiments use all {overview['document_count']:,} saved vectors without metadata filters. The temporary index is removed after the experiment.")
    defaults = overview['defaults']['hnsw']
    query, query_id, qrels_split = query_controls('tuning', 'Experiment query')
    with st.form('tuning_form'):
        columns = st.columns(3)
        backend = columns[0].selectbox('Experiment backend', ['faiss', 'pgvector', 'qdrant'])
        k = columns[1].slider('Experiment Top K', 1, 50, 10)
        repeats = columns[2].slider('Recorded searches', 2, 20, 5)
        columns = st.columns(3)
        m = columns[0].number_input('M', min_value=2, max_value=64, value=defaults['m'], help='Graph connectivity. Higher values generally require more memory.')
        construction = columns[1].number_input('efConstruction', min_value=4, max_value=512, value=defaults['ef_construction'], help='Build exploration breadth. Must be at least 2 × M.')
        ef_search = columns[2].number_input('efSearch', min_value=1, max_value=1000, value=defaults['ef_search'], help='Query exploration breadth. Increase it to explore the recall/latency tradeoff.')
        confirmed = st.checkbox('I confirm creating and removing a temporary experiment index.')
        submitted = st.form_submit_button('Run HNSW experiment', type='primary')
    if submitted:
        if not confirmed:
            st.warning('Confirm the temporary index build before running the experiment.')
        elif not query.strip() or construction < 2 * m:
            st.warning('Enter a query and set efConstruction to at least twice M.')
        else:
            payload = {'query': query, 'query_id': query_id, 'qrels_split': qrels_split, 'backend': backend, 'top_k': k, 'repetitions': repeats, 'confirmed': True,
                       'parameters': {'m': m, 'ef_construction': construction, 'ef_search': ef_search}}
            with st.spinner('Building, measuring, and removing the temporary experiment index…'):
                result = api('POST', '/tune', payload)
            if result is not None:
                history = st.session_state.get('tuning_history', [])
                st.session_state['tuning_history'] = (history + [result])[-20:]
    history = st.session_state.get('tuning_history', [])
    if history:
        result = history[-1]
        st.subheader(f"Latest experiment · {result['backend'].upper()}")
        st.caption(f"Query: {result['query']} · Parameters: {result['parameters']}")
        columns = st.columns(4)
        for column, label, value in zip(columns, ['Exact-neighbor Recall@K', 'Build time', 'P50 search', 'P95 search'],
                                        [f"{result['recall_at_k']:.1%}", f"{result['build_ms']:.2f} ms", f"{result['p50_ms']:.2f} ms", f"{result['p95_ms']:.2f} ms"]):
            column.metric(label, value)
        st.caption(f"{result['repetitions']} measured searches after one warmup. Small samples are exploratory, not stable performance estimates. Recall measures neighbor accuracy over saved vectors, not scientific relevance.")
        search_metrics(result.get('metrics', {}))
        rows = [{'Run': i, 'Backend': row['backend'], **row['parameters'], 'Recall@K': row['recall_at_k'],
                 'Exact P@K': row.get('metrics', {}).get('exact_neighbors', {}).get('precision_at_k'),
                 'Relevance P@K': row.get('metrics', {}).get('relevance', {}).get('precision_at_k'),
                 'Relevance R@K': row.get('metrics', {}).get('relevance', {}).get('recall_at_k'),
                 'Build ms': row['build_ms'], 'P50 ms': row['p50_ms'], 'P95 ms': row['p95_ms']} for i, row in enumerate(history, 1)]
        st.dataframe(rows, hide_index=True, width='stretch')
        st.plotly_chart(px.scatter(rows, x='P50 ms', y='Recall@K', color='Backend', hover_data=['Run', 'm', 'ef_construction', 'ef_search']), width='stretch')
        with st.expander('Measured samples and backend details'):
            st.json({'samples_ms': result['latency_samples_ms'], 'details': result['details'], 'timing_scope': result['timing_scope']})
        st.caption('Build scopes differ: FAISS vector addition, PostgreSQL CREATE INDEX, Qdrant point upload through indexed readiness. No disk, memory, or CPU isolation is provided.')
        st.download_button('Download experiment history', json.dumps(history, indent=2), 'hnsw-experiments.json', 'application/json')
