{% docs stg_orders_desc %}
Staging table for customer orders used as the cleaned source for analytics.
{% enddocs %}

{% docs order_id_desc %}
Unique identifier for the order, stable across system retries.
{% enddocs %}

{% docs created_at_desc %}
Timestamp when the order was created in the source system.
{% enddocs %}

{% docs stg_customers_desc %}
Customer records used for analytics and downstream reporting.
{% enddocs %}

{% docs email_desc %}
Email address for the customer used for communication and login.
{% enddocs %}

{% docs marketing_opt_in_desc %}
Whether the customer opted in to marketing communications.
{% enddocs %}
