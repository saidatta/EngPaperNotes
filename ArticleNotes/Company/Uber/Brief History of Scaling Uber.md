#### Overview
- Uber, conceived in Paris in 2008, has evolved into the world's largest mobility platform, operating in over 70 countries and 10,500 cities. Uber Eats, launched in 45 countries, is the largest food delivery platform outside China.
#### Initial Growth and Challenges
- **2009**: Uber's first version, built by contractors using the LAMP stack, was limited by major concurrency issues.
- **Circa 2011**: Uber adopted Node.js for real-time needs and Python for business logic, addressing concurrency and the need for real-time data processing.
#### Transition to a Service Oriented Architecture
- **Two-Service Architecture**: A Node.js service for dispatch (connected to MongoDB/Redis) and a Python service for API functions (connected to PostgreSQL).
- **Object Node ("ON") Layer**: Introduced for resilience in the dispatch flow.
#### From Monoliths to Microservices
- **Circa 2013**: Uber transitioned to a microservice architecture to manage growing features and team sizes, leading to about 100 services by 2014.
- **Operational Complexity**: Addressed with internal tools like Clay for service frameworks, TChannel over Hyperbahn for service resilience, Apache Thrift for RPC interfaces, Flipr for feature flagging, and M3 for observability.
#### Scaling the Trip Database
- **Circa 2014**: A single PostgreSQL database for trips became a bottleneck.
- **Schemaless**: An append-only sparse three-dimensional persistent hash map built on MySQL was developed for trip data, enabling horizontal scaling.
#### Dispatch Service Evolution
- **Splitting Dispatch**: The monolithic dispatch service was divided into a real-time API gateway (RTAPI) and a dispatch service.
- **Ringpop**: Developed for sharing geospatial and supply positioning information.
#### Mobile App Development
- **Circa 2016**: Introduction of the RIB architecture for mobile app development, facilitating clear separation of responsibilities and scaling to multiple engineering teams.
#### Launch and Scaling of Uber Eats
- **Circa 2017**: Uber Eats leveraged existing Uber technology, quickly growing to support various ordering modes and marketplace configurations.
#### Standardization and Efficiency: Project Ark
- **Circa 2018**: Project Ark was initiated to standardize engineering practices, reducing the number of microservices, code repositories, and establishing standard programming languages and architectural layers.
#### Modernizing the Gateway
- **Circa 2020**: A new Edge Gateway was developed, standardizing API lifecycle management and segregating responsibilities into distinct layers (Edge, Presentation, Product, Domain).

#### Rewriting the Fulfillment Platform
- **Circa 2021**: The Fulfillment Platform was rewritten, using Google Cloud Spanner for transactional consistency, scalability, and low operational overhead.

#### Continuous Improvements and Innovations
- Uber has constantly evolved its systems, including API gateway, fulfillment stack, money stack, and data intelligence platforms. Tools like Cadence, Apache Hudi, Redis caches, and data infrastructure like Presto and Spark were leveraged for growth.

#### Future: Migration to the Cloud
- **Current Focus**: Uber is working on migrating a significant portion of its server fleet to the cloud, having already made 4000 stateless microservices portable and setting up Project Crane for this transition.

#### Sources
- [Josh Clemm, Senior Director of Engineering, Uber Eats](https://www.linkedin.com/in/joshclemm/) (January 11, 2024)
- [Uber's Engineering Blog](https://eng.uber.com/)