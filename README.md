# piscine_python_ml

## Bootcamp Python

```bash
# For bootcamp python
export PYTHONPATH=/home/rcutte/Desktop/piscine_python_ml/bootcamp_python/
```

### Types

<details>
<summary>More infos</summary>

- 🐍 [ Python - Built-in Types](https://docs.python.org/3/library/stdtypes.html)
  - https://docs.python.org/3/library/stdtypes.html#truth-value-testing
  - https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex
  - https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex
  - https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range
  - https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str
  - https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset
  - https://docs.python.org/3/library/stdtypes.html#mapping-types-dict

```mermaid
classDiagram
    class object {
        <<built-in>>
    }
    class int {
        <<built-in>>
    }
    class float {
        <<built-in>>
    }
    class str {
        <<built-in>>
    }
    class list {
        <<built-in>>
    }
    class dict {
        <<built-in>>
    }
    class tuple {
        <<built-in>>
    }
    class set {
        <<built-in>>
    }

    object <|-- int
    object <|-- float
    object <|-- complex
    object <|-- str
    object <|-- list
    object <|-- dict
    object <|-- tuple
    object <|-- set

    note for object "Base class of all Python objects"
    note for int "Whole numbers (e.g., 1, 2, 3)"
    note for float "Decimal numbers (e.g., 3.14, -0.5)"
    note for complex "Complex numbers (e.g., 3+4j)"
    note for str "Text strings (e.g., 'hello', \'\'world\'\')"
    note for list "Ordered collections (e.g., [1, 2, 3])"
    note for dict "Key-value mappings (e.g., {'a': 1})"
    note for tuple "Immutable ordered collections (e.g., (1, 2, 3))"
    note for set "Unordered unique elements (e.g., {1, 2, 3})"
```

</details>

#### Dict

<details>
<summary>More infos</summary>

- 🐍 [ Python - Dictionaries ](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)

```mermaid
flowchart LR
    classDef basic fill:#90EE90,stroke:#006400,color:#000000
    classDef modify fill:#FFB6C1,stroke:#8B0000,color:#000000
    classDef query fill:#ADD8E6,stroke:#000080,color:#000000

    Start["Dictionary Operations"] --> Basic["Basic Operations"]
    Start --> Modify["Modification"]
    Start --> Query["Query Operations"]

    subgraph "Basic Operations"
        Basic --> Create["Creation
        d = {}"]
        Basic --> Access["Access
        d[key]"]
        Basic --> Check["Check Existence
        key in d"]
    end

    subgraph "Modification"
        Modify --> Add["Add/Update
        d[key] = value"]
        Modify --> Delete["Delete
        del d[key]<br>d.pop(key, return_value)"]
        Modify --> Clear["Clear All
        d.clear()"]
    end

    subgraph "Query Operations"
        Query --> Keys["Get Keys
        d.keys()"]
        Query --> Values["Get Values
        d.values()"]
        Query --> Items["Get Items
        d.items()"]
    end

    class Basic,Create,Access,Check basic
    class Modify,Add,Delete,Pop,Clear modify
    class Query,Keys,Values,Items query
```

</details>

### Formating

<details>
<summary>More infos</summary>

- 🐍 [ Python - Format Specification Mini-Language ](https://docs.python.org/3.9/library/string.html#format-specification-mini-language)

```mermaid
flowchart LR
    classDef basic fill:#90EE90,stroke:#006400,color:#000000
    classDef advanced fill:#FFB6C1,stroke:#8B0000,color:#000000
    classDef output fill:#ADD8E6,stroke:#000080,color:#000000

    Start["String Formatting"] --> Basic["Basic Methods"]
    Start --> Advanced["Advanced Methods<br><br>[[fill]align][sign][#][0][width][grouping_option][.precision][type]"]

    subgraph "Basic Methods"
        Basic --> F["f-strings
        name = 'John'
        f'Hello, {name}!'"]
        Basic --> Format["str.format()
        'Hello, {}!'.format(name)"]
        Basic --> Percent["% Operator
        'Hello, %s!' % name"]
    end

    subgraph "Advanced Methods"
        Advanced --> Align["Alignment
        '{:-^10}'.format(name)"]
        Advanced --> Fill["Fill Character
        '{:_>10}'.format(name)"]
        Advanced --> Width["Width Specifier
        '{:10}'.format(name)"]
        Advanced --> Precision["Precision
        '{:.2f}'.format(3.14159)"]
    end

    F --> Output1["Output:
    Hello, John!"]
    Format --> Output2["Output:
    Hello, John!"]
    Percent --> Output3["Output:
    Hello, John!"]
    Fill --> Output4["Output:<br>______John"]
    Align --> Output5["Output:<br>---John---"]
    Width --> Output6["Output:
    John"]
    Precision --> Output7["Output:
    3.14"]

    class Basic,Format,F,Percent basic
    class Advanced,Align,Width,Precision,Fill advanced
    class Output1,Output2,Output3,Output4,Output5,Output6,Output7 output
```

</details>

### Vectors

[ 📹 Youtube - Vectors - Essence of linear algebra](https://youtu.be/fNk_zzaMoSs?si=nukJqaKyoSkP-tFA)
🐍 [ Python - datamodel - numeric types](https://docs.python.org/3.9/reference/datamodel.html#emulating-numeric-types)

### Builtins functions

- 🐍 [ Python - Builtin Functions](https://docs.python.org/3.9/library/functions.html)
  - https://docs.python.org/3.9/library/functions.html#vars
  - https://docs.python.org/3.9/library/functions.html#dir

### Decorators

<details>
<summary>Decorators</summary>

- 🐍 [ Python - Decorators](https://docs.python.org/3/glossary.html#term-decorator)

```mermaid
sequenceDiagram
    participant C as Client Code
    participant D as Decorator (@my_decorator)
    participant W as Wrapper Function
    participant O as Original Function

    Note over C,O: Normal Execution Flow
    C->>+D: Call decorated function
    D->>+W: Execute wrapper
    Note over W: Before function code runs
    W->>+O: Call original function
    O-->>-W: Return from original
    Note over W: After function code runs
    W-->>-D: Return to decorator
    D-->>-C: Final return to client

    Note over C,O: Equivalent Manual Decoration
    C->>D: my_decorator(original_function)
    D-->>C: Returns decorated function
```

Best Practices

1. Always use functools.wraps to preserve the original function's metadata:

```python
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Preserves function name, docstring, etc.
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

2. Handle arguments properly using \*args and \*\*kwargs:

```python
def flexible_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Received args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper
```

</details>

### Context Managers

<details>
<summary>More infos</summary>

- 🐍 [ Python - Context Managers](https://docs.python.org/3/library/stdtypes.html#typecontextmanager)

```mermaid
sequenceDiagram
    participant C as Client Code
    participant W as With Statement
    participant CM as Context Manager

    Note over C,CM: Normal Execution
    C->>W: Enter with block
    W->>CM: __enter__()
    CM-->>W: Return value
    W->>C: Assign to 'as' variable
    Note over C: Execute block content
    C->>W: Block complete
    W->>CM: __exit__(None, None, None)

    Note over C,CM: Exception Case
    C->>W: Enter with block
    W->>CM: __enter__()
    CM-->>W: Return value
    W->>C: Assign to 'as' variable
    Note over C: Execute block content
    C->>W: Raise Exception
    W->>CM: __exit__(exc_type, exc_val, traceback)
    alt __exit__ returns True
        CM-->>W: Suppress exception
    else __exit__ returns False
        CM-->>W: Propagate exception
    end
```

</details>

### Package

🐍 [Python - Creating a packgage](https://docs.python.org/3.9/distributing/index.html) - https://packaging.python.org/en/latest/tutorials/packaging-projects/#packaging-python-projects - https://docs.python.org/3/tutorial/modules.html#packages - [Setuptools - Config](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)

```bash
# Setup the env
python3 -m venv .venv
source .venv/bin/activate
```

```bash
pip install setuptools wheel twine

# Setup your package
```

```bash
# After setting up the package - update build
python3 -m pip install --upgrade build
```

```bash
# Build the package
python3 -m build
```

📜 [Choose a license](https://choosealicense.com/)

### Numpy

- https://numpy.org/doc/2.2/
- https://numpy.org/doc/2.2/user/absolute_beginners.html
- https://numpy.org/doc/2.2/reference/index.html
- https://numpy.org/doc/2.2/user/basics.broadcasting.html#basics-broadcasting

### Matplotlib

https://matplotlib.org/stable/contents.html
https://matplotlib.org/stable/users/explain/quick_start.html

#### Plot Setup

https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter

##### Plot Colors and markers

https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
https://matplotlib.org/stable/users/explain/colors/colormaps.html#sphx-glr-users-explain-colors-colormaps-py

### Pandas

- https://pandas.pydata.org/docs/getting_started/index.html#intro-to-pandas
- https://pandas.pydata.org/docs/user_guide/10min.html#min
- https://pandas.pydata.org/docs/
- https://pandas.pydata.org/docs/user_guide/index.html

## Math

```mermaid
---
title: Mathematical Concepts
config:
    theme: default
---
mindmap
    root((Mathematical<br/>Concepts))
        Linear Algebra
            Vectors
            Matrices
            Transformations
            Eigenvalues
            Determinants
        Descriptive Statistics
            Central Tendency
                Mean
                Median
                Mode
            Dispersion
                Variance
                Standard Deviation
                Quartiles
            Shape Description
                Skewness
                Kurtosis
```

### Visualization

- [3D Geogebra](https://www.geogebra.org/3d)

### Statistics

#### Central Tendency

- Mean: https://www.mathsisfun.com/mean.html
- Median: https://www.mathsisfun.com/median.html

##### K-Means

https://neptune.ai/blog/k-means-clustering
https://www.geeksforgeeks.org/k-means-clustering-introduction/
[![K-Means](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif)](https://en.wikipedia.org/wiki/K-means_clustering)

#### Dispersion

- Quartiles: https://www.mathsisfun.com/data/quartiles.html

![Quartiles](https://www.mathsisfun.com/data/images/interquartile-range.svg)

- Standard Deviation and Variance: https://www.mathsisfun.com/data/standard-deviation.html

![Standard Deviation](https://www.mathsisfun.com/data/images/statistics-standard-deviation.gif)

### Linear Algebra

#### L1 and L2

- L1 - Manhattan Distance
    - Also: Taxicab Distance
    - Measures the distance between two points as if you were traveling along a city grid and can only move along the streets (no diagonals).
    - May be preferred when dimensions are not of the same scale.
- L2 - Euclidean Distance
    - Measures the distance between two points as if you could travel through the air (no obstacles).
    - More sensitive to differences in magnitude between dimensions.

[Difference betwenn L1 and L2](https://medium.com/@datasciencejourney100_83560/difference-between-l1-manhattan-and-l2-euclidean-distance-c70b5da25fe0)

![L1 and L2](https://miro.medium.com/v2/resize:fit:720/format:webp/1*LIoPQufLTF8xDgukFTej_g.png)

#### Vectors

[ 📹 Youtube - Vectors - Essence of linear algebra](https://youtu.be/fNk_zzaMoSs?si=nukJqaKyoSkP-tFA)

#### Matrices

[ 📹 Youtube - Matrices - Essence of linear algebra](https://youtu.be/kYB8IZa5AuE?si=3Q9Q9QJvQ7qQ7qQ7)
