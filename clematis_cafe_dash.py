
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
customer_features = pd.read_csv('engineered_customer_features.csv')
data_clean = pd.read_csv('cleaned_clematis_cafe_data.csv')
model_data = pd.read_csv('engineered_customer_features.csv').merge(
    pd.read_csv('cleaned_clematis_cafe_data.csv')[['CustomerID', 'LoyaltyMember']].drop_duplicates(),
    on='CustomerID'
)

# Prepare visuals
monthly_visits = data_clean.groupby(data_clean['Month']).size().reset_index(name='Transactions')
y = model_data['LoyaltyMember']
X = model_data[['AvgSpendPerVisit', 'TotalSpend', 'TotalVisits', 'LoyaltyScore', 'CustomerSegment']]
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Clematis Cafe - Customer Insights Dashboard'),

    html.Div([
        html.Label('Select Customer Segment:'),
        dcc.Dropdown(
            id='segment-dropdown',
            options=[
                {'label': f'Segment {seg}', 'value': seg} for seg in sorted(customer_features['CustomerSegment'].unique())
            ],
            value=0
        ),
        dcc.Graph(id='segment-graph')
    ]),

    html.Div([
        dcc.Graph(id='loyalty-graph')
    ]),

    html.Div([
        dcc.Graph(
            id='monthly-graph',
            figure=px.line(
                monthly_visits,
                x='Month',
                y='Transactions',
                markers=True,
                title='Monthly Transactions Trend'
            )
        )
    ]),

    html.Div([
        html.H2('Predict Loyalty Signup'),
        html.Label('Avg Spend Per Visit:'),
        dcc.Input(id='input-avg-spend', type='number', value=10),
        html.Label('Total Spend:'),
        dcc.Input(id='input-total-spend', type='number', value=100),
        html.Label('Total Visits:'),
        dcc.Input(id='input-total-visits', type='number', value=10),
        html.Label('Loyalty Score:'),
        dcc.Input(id='input-loyalty-score', type='number', value=80),
        html.Label('Customer Segment:'),
        dcc.Input(id='input-segment', type='number', value=0),
        html.Button('Predict', id='predict-button', n_clicks=0),
        html.Div(id='prediction-result')
    ])
])

@app.callback(
    Output('segment-graph', 'figure'),
    Input('segment-dropdown', 'value')
)
def update_segment_graph(selected_segment):
    filtered_data = customer_features[customer_features['CustomerSegment'] == selected_segment]
    fig = px.scatter(
        filtered_data,
        x='TotalVisits',
        y='TotalSpend',
        color='CustomerSegment',
        title=f'Customer Segment {selected_segment}: Spend vs Visits'
    )
    return fig

@app.callback(
    Output('loyalty-graph', 'figure'),
    Input('segment-dropdown', 'value')
)
def update_loyalty_graph(selected_segment):
    segment_customers = customer_features[customer_features['CustomerSegment'] == selected_segment]['CustomerID']
    loyalty_spend_segment = model_data[model_data['CustomerID'].isin(segment_customers)]
    loyalty_spend_summary = loyalty_spend_segment.groupby('LoyaltyMember')['TotalSpend'].mean().reset_index()
    loyalty_spend_summary['LoyaltyMember'] = loyalty_spend_summary['LoyaltyMember'].map({0: 'Non-Member', 1: 'Member'})
    fig = px.bar(
        loyalty_spend_summary,
        x='LoyaltyMember',
        y='TotalSpend',
        title=f'Avg Spend: Loyalty vs Non-Loyalty (Segment {selected_segment})'
    )
    return fig

@app.callback(
    Output('prediction-result', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-avg-spend', 'value'),
    State('input-total-spend', 'value'),
    State('input-total-visits', 'value'),
    State('input-loyalty-score', 'value'),
    State('input-segment', 'value')
)
def predict_loyalty(n_clicks, avg_spend, total_spend, total_visits, loyalty_score, segment):
    if n_clicks > 0:
        input_features = [[avg_spend, total_spend, total_visits, loyalty_score, segment]]
        prediction = model.predict(input_features)[0]
        if prediction == 1:
            return '✅ Prediction: Likely to Join Loyalty Program'
        else:
            return '⚠️ Prediction: Unlikely to Join Loyalty Program'
    return ''

if __name__ == '__main__':
    app.run_server(debug=True)
