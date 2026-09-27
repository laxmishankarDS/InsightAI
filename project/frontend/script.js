// ==========================================================
// INSIGHTAI DASHBOARD
// ==========================================================

const API_URL = "http://127.0.0.1:8000";


// ==========================================================
// CHART VARIABLES
// ==========================================================

let salesChart = null;
let categoryChart = null;
let regionChart = null;


// ==========================================================
// FORMAT CURRENCY
// ==========================================================

function formatCurrency(value) {

    return Number(value).toLocaleString("en-IN", {

        style: "currency",

        currency: "INR",

        maximumFractionDigits: 0

    });

}


// ==========================================================
// FORMAT NUMBER
// ==========================================================

function formatNumber(value) {

    return Number(value).toLocaleString("en-IN");

}


// ==========================================================
// GET FILTERS
// ==========================================================

function getFilters() {

    const category =
        document.getElementById("categoryFilter").value;

    const region =
        document.getElementById("regionFilter").value;

    const product =
        document.getElementById("productFilter").value;


    return {

        category,
        region,
        product

    };

}


// ==========================================================
// BUILD QUERY
// ==========================================================

function buildQuery(filters) {

    const params = new URLSearchParams();


    if (filters.category) {

        params.append(
            "category",
            filters.category
        );

    }


    if (filters.region) {

        params.append(
            "region",
            filters.region
        );

    }


    if (filters.product) {

        params.append(
            "product",
            filters.product
        );

    }


    return params.toString();

}


// ==========================================================
// LOAD FILTER OPTIONS
// ==========================================================

async function loadFilters() {

    try {

        const response =
            await fetch(`${API_URL}/api/filters`);

        const data =
            await response.json();


        const categoryFilter =
            document.getElementById(
                "categoryFilter"
            );

        const regionFilter =
            document.getElementById(
                "regionFilter"
            );

        const productFilter =
            document.getElementById(
                "productFilter"
            );


        data.categories.forEach(category => {

            const option =
                document.createElement("option");

            option.value = category;

            option.textContent = category;

            categoryFilter.appendChild(option);

        });


        data.regions.forEach(region => {

            const option =
                document.createElement("option");

            option.value = region;

            option.textContent = region;

            regionFilter.appendChild(option);

        });


        data.products.forEach(product => {

            const option =
                document.createElement("option");

            option.value = product;

            option.textContent = product;

            productFilter.appendChild(option);

        });


    }

    catch (error) {

        console.error(
            "Filter loading error:",
            error
        );

    }

}


// ==========================================================
// LOAD SUMMARY
// ==========================================================

async function loadSummary(filters) {

    const query =
        buildQuery(filters);


    const response =
        await fetch(
            `${API_URL}/api/summary?${query}`
        );


    const data =
        await response.json();


    document.getElementById("revenue")
        .textContent =
        formatCurrency(data.revenue);


    document.getElementById("profit")
        .textContent =
        formatCurrency(data.profit);


    document.getElementById("orders")
        .textContent =
        formatNumber(data.orders);


    document.getElementById("quantity")
        .textContent =
        formatNumber(data.quantity);

}


// ==========================================================
// LOAD MONTHLY CHART
// ==========================================================

async function loadMonthlyChart(filters) {

    const query =
        buildQuery(filters);


    const response =
        await fetch(
            `${API_URL}/api/monthly?${query}`
        );


    const data =
        await response.json();


    const labels =
        data.map(item => item.Month);


    const revenue =
        data.map(item => item.Revenue);


    const profit =
        data.map(item => item.Profit);


    if (salesChart) {

        salesChart.destroy();

    }


    salesChart = new Chart(

        document.getElementById(
            "salesChart"
        ),

        {

            type: "line",

            data: {

                labels,

                datasets: [

                    {

                        label: "Revenue",

                        data: revenue,

                        borderColor:
                            "#2563eb",

                        backgroundColor:
                            "rgba(37,99,235,0.08)",

                        borderWidth: 3,

                        fill: true,

                        tension: 0.35

                    },

                    {

                        label: "Profit",

                        data: profit,

                        borderColor:
                            "#10b981",

                        backgroundColor:
                            "rgba(16,185,129,0.05)",

                        borderWidth: 3,

                        fill: true,

                        tension: 0.35

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                interaction: {

                    mode: "index",

                    intersect: false

                },

                plugins: {

                    legend: {

                        position: "top"

                    }

                },

                scales: {

                    y: {

                        beginAtZero: true,

                        ticks: {

                            callback: function(value) {

                                return "₹" +
                                    Number(value)
                                    .toLocaleString(
                                        "en-IN"
                                    );

                            }

                        }

                    }

                }

            }

        }

    );

}


// ==========================================================
// CATEGORY CHART
// ==========================================================

async function loadCategoryChart(filters) {

    const query =
        buildQuery(filters);


    const response =
        await fetch(
            `${API_URL}/api/categories?${query}`
        );


    const data =
        await response.json();


    const labels =
        data.map(item => item.Category);


    const values =
        data.map(item => item.Revenue);


    if (categoryChart) {

        categoryChart.destroy();

    }


    categoryChart = new Chart(

        document.getElementById(
            "categoryChart"
        ),

        {

            type: "doughnut",

            data: {

                labels,

                datasets: [

                    {

                        data: values,

                        backgroundColor: [

                            "#2563eb",
                            "#06b6d4",
                            "#10b981",
                            "#f59e0b",
                            "#ef4444",
                            "#8b5cf6",
                            "#ec4899"

                        ],

                        borderWidth: 0

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {

                        position: "bottom"

                    }

                }

            }

        }

    );

}


// ==========================================================
// TOP PRODUCTS TABLE
// ==========================================================

async function loadProducts(filters) {

    const query =
        buildQuery(filters);


    const response =
        await fetch(
            `${API_URL}/api/products?${query}`
        );


    const data =
        await response.json();


    const table =
        document.getElementById(
            "productTable"
        );


    table.innerHTML = "";


    data.forEach(item => {

        const row =
            document.createElement("tr");


        row.innerHTML = `

            <td>
                <strong>
                    ${item.Product_Name}
                </strong>
            </td>

            <td>
                ${formatCurrency(item.Revenue)}
            </td>

            <td>
                ${formatNumber(item.Quantity)}
            </td>

            <td>
                ${formatCurrency(item.Profit)}
            </td>

        `;


        table.appendChild(row);

    });

}


// ==========================================================
// REGION CHART
// ==========================================================

async function loadRegionChart(filters) {

    const query =
        buildQuery(filters);


    const response =
        await fetch(
            `${API_URL}/api/regions?${query}`
        );


    const data =
        await response.json();


    const labels =
        data.map(item => item.Region);


    const values =
        data.map(item => item.Revenue);


    if (regionChart) {

        regionChart.destroy();

    }


    regionChart = new Chart(

        document.getElementById(
            "regionChart"
        ),

        {

            type: "bar",

            data: {

                labels,

                datasets: [

                    {

                        label: "Revenue",

                        data: values,

                        backgroundColor:
                            "#2563eb",

                        borderRadius: 8

                    }

                ]

            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {

                        display: false

                    }

                },

                scales: {

                    y: {

                        beginAtZero: true,

                        ticks: {

                            callback: function(value) {

                                return "₹" +
                                    Number(value)
                                    .toLocaleString(
                                        "en-IN"
                                    );

                            }

                        }

                    }

                }

            }

        }

    );

}


// ==========================================================
// LOAD EVERYTHING
// ==========================================================

async function loadDashboard() {

    try {

        const filters =
            getFilters();


        await Promise.all([

            loadSummary(filters),

            loadMonthlyChart(filters),

            loadCategoryChart(filters),

            loadProducts(filters),

            loadRegionChart(filters)

        ]);


        document.getElementById(
            "lastUpdated"
        ).textContent =
            "Last updated: " +
            new Date().toLocaleTimeString(
                "en-IN"
            );


        console.log(
            "InsightAI dashboard updated successfully."
        );

    }

    catch (error) {

        console.error(
            "Dashboard loading error:",
            error
        );

        alert(
            "Unable to connect to InsightAI backend. Make sure FastAPI is running."
        );

    }

}


// ==========================================================
// RESET FILTERS
// ==========================================================

function resetFilters() {

    document.getElementById(
        "categoryFilter"
    ).value = "";


    document.getElementById(
        "regionFilter"
    ).value = "";


    document.getElementById(
        "productFilter"
    ).value = "";


    loadDashboard();

}


// ==========================================================
// START DASHBOARD
// ==========================================================

document.addEventListener(
    "DOMContentLoaded",
    async function() {

        await loadFilters();

        await loadDashboard();

    }
);
