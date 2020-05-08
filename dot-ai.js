class Action {
    x = 0
    nx = 0
    x = 0
    ny = 0
    res= ''
    props = {}
    step = 0

    constructor(props) {
        this.props = props
        this.setProp(props)
    }

    setProp(props) {
        Object.keys(props).forEach(key => {
            this[key] = props[key]
        })
    }

    get q() {
        let _q = 0
        const _zeta = this.zeta
        const condition = {
            1: _zeta >= 0 && _zeta <= 90,
            2: _zeta > 90 && _zeta <= 180,
            3: _zeta > 180 && _zeta <= 270,
            4: _zeta > 270 && _zeta <= 360,
        }

        Object.keys(condition).forEach(key => {
            if (condition[key]) {
                _q = key
            }
        })
        // console.log(_q, _zeta, condition[_q] )
        return _q
    }

    get diff_x () { return this.nx - this.x }
    get diff_y () { return this.ny - this.y }

    get dist() {
        return (Math.sqrt(Math.pow(this.x - this.nx, 2) + Math.pow(this.y - this.ny, 2)))
    }

    get zeta() {
        const y = Math.abs(this.y - this.ny)
        const x =  Math.abs(this.x - this.nx)
        if (y == 0 || y == 0) {
            return 0
        }
        return (Math.atan((y)/(x)))
    }

    get slope() {
        return this.diff_y == 0 || this.diff_x == 0 ? 0 : this.diff_y/this.diff_x
    }

    get tensor() {
        const g = (value) => {
            if (value == 0) { return 0 }
            else if (value > 0) { return 1  }
            else if (value < 0) { return -1 }
        }

        const l = (value) => {
            // if (value <= 40) { return value }
            if (value <= 1) { return 'L0' }
            else if (value <= 2.5) { return 'L0.25' }
            else if (value <= 5) { return 'L0.5' }
            else if (value <= 10) { return 'L1' }
            else if (value <= 20) { return 'L2' }
            else if (value <= 25) { return 'L2.5' }
            else if (value <= 30) { return 'L3' }
            else if (value <= 50) { return 'L3' }
            else if (value <= 70) { return 'L7' }
            else if (value <= 110) { return 'L110' }
            else if (value <= 130) { return 'L130' }
            else { return 'OUT'  }
        }

        const q = (point)  => {
            if (point.join(',') === [1, 1].join(',')) { return 'Q1' }
            else if (point.join(',') === [-1,1].join(',')) { return 'Q2' }
            else if (point.join(',') === [-1,-1].join(',')) { return 'Q3' }
            else if (point.join(',') === [1,-1].join(',')) { return 'Q4' }

            if (point.join(',') === [0, 1].join(',')) { return 'R' }
            else if (point.join(',') === [1,0].join(',')) { return 'F' }
            else if (point.join(',') === [0,-1].join(',')) { return 'L' }
            else if (point.join(',') === [-1,0].join(',')) { return 'B' }
        }

        const data = [
            g(this.diff_x),
            g(this.diff_y),
            parseFloat(this.slope.toFixed(2)),
            parseFloat(this.zeta.toFixed(2)),
        ]

        console.log('mark', data)
       
        return tf.tensor(data)
    }
    toTensor_step() {
        const g = (value) => {
            if (value == 0) { return 0 }
            else if (value > 0) { return 1  }
            else if (value < 0) { return -1 }
        }
        const q = (point)  => {
            if (point.join(',') === [1, 1].join(',')) { return 'Q1' }
            else if (point.join(',') === [-1,1].join(',')) { return 'Q2' }
            else if (point.join(',') === [-1,-1].join(',')) { return 'Q3' }
            else if (point.join(',') === [1,-1].join(',')) { return 'Q4' }

            if (point.join(',') === [0, 1].join(',')) { return 'R' }
            else if (point.join(',') === [1,0].join(',')) { return 'F' }
            else if (point.join(',') === [0,-1].join(',')) { return 'L' }
            else if (point.join(',') === [-1,0].join(',')) { return 'B' }
        }

        const l = (value) => {
            value = Math.abs(value)
            // if (value <= 40) { return value }
            if (value <= 1) { return 'L0' }
            else if (value <= 2.5) { return 'L0.25' }
            else if (value <= 5) { return 'L0.5' }
            else if (value <= 10) { return 'L1' }
            else if (value <= 20) { return 'L2' }
            else if (value <= 25) { return 'L2.5' }
            else if (value <= 30) { return 'L3' }
            else if (value <= 50) { return 'L3' }
            else if (value <= 70) { return 'L7' }
            else if (value <= 110) { return 'L110' }
            else if (value <= 130) { return 'L130' }
            else { return 'OUT'  }
        }

        const data = [
            parseFloat(this.dist.toFixed(2)),
            this.res,
            l(this.dist),
        ]
        // console.log('walk-step toTensor_step', data)
        return tf.tensor(data)
    }
  }

data = [
    new Action({ y: 0, x: 0, res: 'x', step: 0 }),
];

const classifier_save = (classifier, localStorageKey) => {
    const _classifier = classifier
    // Save it to a string:
    let str = JSON.stringify(
        Object.entries(_classifier.getClassifierDataset())
            .map(([label, data])=>[label, Array.from(data.dataSync()), data.shape])
    )

    localStorage.setItem(localStorageKey, str);
}

const classifier_load = (localStorageKey) => {
    _classifier = knnClassifier.create();

    var str = localStorage.getItem(localStorageKey);
    if (str == null) {
        return _classifier;
    }
    // Load it back into a fresh classifier:
    _classifier.setClassifierDataset(
        Object.fromEntries(
            JSON.parse(str).map(([label, data, shape])=>[label, tf.tensor(data, shape)])
        ) 
    );

    return _classifier;
}

let addExample = 0
let eat_food = 0
let classifier  = null;

const init = async function() {
    // Create the classifier.
    classifier = classifier_load('classifier-nav');
    classifierStep = classifier_load('classifier-step');


    // Add MobileNet activations to the model repeatedly for all classes.
    data.forEach(act => {
        classifier.addExample(act.tensor, act.res);
        classifierStep.addExample(act.toTensor_step(), act.step);
    })
}

init();

function makeid(length) {
    var result           = '';
    var characters       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    var charactersLength = characters.length;
    for ( var i = 0; i < length; i++ ) {
       result += characters.charAt(Math.floor(Math.random() * charactersLength));
    }
    return result;
 }

class People
{
    use_ai = false
    main_color = ''
    id = 0
    _x = 0
    nx = 0
    ny = 0
    _y = 0
    step = 0
    color = 'red'
    radius = 1
    canvas =  null
    nav = 'w'
    speed = 2.5
    _isDanger = false
    score = 0

    constructor (radius, canvas) {
        this.id = makeid(8)
        this.canvas = canvas
        this.x = parseInt(Math.random() * (canvas.width - radius))
        this.y = parseInt(Math.random() * (canvas.height - radius))
        this.radius = radius
    }

    get isDanger(){ return this._isDanger; }

    set isDanger(value) {
        if (value) {
            this.step = 0
            this.think()
        }
        this._isDanger = value
    }

    get y(){ return this._y; }

    set y(value) {
        if (
            value - this.radius > 0 &&
            value + this.radius < this.canvas.height
        ) {
            this._y = value
            this.isDanger = false
            return;
        }
        this._y = (value > this.canvas.height / 2 ? this.canvas.height - this.radius - 1 : this.radius + 1)
        this.isDanger = true
        return;
    }

    set x(value) {
        if (
            value - this.radius > 0 && 
            value + this.radius < this.canvas.width
        ) {
            this._x = value
            this.isDanger = false
            return;
        }
        this._x = (value > this.canvas.width / 2 ? (this.canvas.width - this.radius - 1) : this.radius + 1)
        this.isDanger = true
        return;
    }

    get x() { return this._x }

    async think(foods, callback) {
        const pos = { x: this.x, y: this.y };
        const control = [
            'w', 'a', 's', 'd',
            'wa', 'wd',
            'as', 'ds',
        ]
        const steps = Array.from({length: 10}, (v, k) => k+1); 

        if (this.step <= 0) {
            let nextNav = control[parseInt(Math.random() * control.length)]
            let nextStep = steps[parseInt(Math.random() * steps.length)]
            // let nextStep = 1
            // this.step = 5
            // console.log(this.color,'random',{ nav: this.nav })
            if (foods) {
                let food_min_dist = foods.filter(f => f.constructor.name == Food.name)
                    // .filter(f => (f.x >= pos.x - 20) && (f.x <= pos.x + 20))
                    // .filter(f => (f.y >= pos.y - 20) && (f.y <= pos.y + 20))
                    .sort((a, b) => {
                        const dist_a = Math.sqrt(Math.pow(pos.x - a.x, 2) + Math.pow( pos.y - a.y, 2))
                        const dist_b = Math.sqrt(Math.pow(pos.x - b.x, 2) + Math.pow( pos.y - b.y, 2))
                        return dist_a - dist_b
                    })[0]
        
                if (food_min_dist) {
                    this.target = food_min_dist
                    console.warn('predictStep-pre', { nextStep, nextNav });
                   
                    nextNav = await this.predictNav(food_min_dist, nextNav, pos);
                    nextStep = await this.predictStep(food_min_dist, nextNav, nextStep, pos);
                    console.warn('predictStep-post', { nextStep, nextNav });
                }
            }
            this.nav = nextNav
            this.step = nextStep
        }
    }

    async predictStep (food_min_dist, nav, step) {
        let _nav = nav
        let _step = Math.round(step)
        const food = food_min_dist

        const action = new Action({ x: food.x, y: food.y, nx: this.x, ny: this.y });
        let result = null
        try {
            result = await classifierStep.predictClass(action.toTensor_step());
        } catch (error) {
            classifierStep.addExample(action.toTensor_step(), Math.ceil(_step) );
            result = await classifierStep.predictClass(action.toTensor_step());
        }

        if (this.use_ai) {
            return result.label
        }

        console.log('predictStep', { nav: this.nav, step: result.label, _step, _nav });

        const dist_by_random = ((() => { 
           let x = this.x
           let y = this.y
            for (let index = 0; index < _step; index++) {
                const poin = this.nextPoint(_nav, x, y)
                x = poin.x
                y = poin.y
            }

            let dist = Math.pow(x - food.x, 2) + Math.pow( y - food.y, 2)
            dist = Math.sqrt(dist);
            return dist
        })()); 
        
        const dist_by_ml = ((() => { 
            let x = this.x
            let y = this.y
            for (let index = 0; index < Number(result.label); index++) {
                const poin = this.nextPoint(_nav, x, y)
                x = poin.x
                y = poin.y
            }

            let dist = Math.pow(x - food.x, 2) + Math.pow( y - food.y, 2)
            dist = Math.sqrt(dist);
            return dist
        })()); 

        console.log(this.color, 'predictStep', 'walk-step', {dist_by_ml, dist_by_random}, { _step, ml_step: result.label, res: nav });
        if (dist_by_ml < dist_by_random  &&  result.label != '0') {
            _step = result.label;
            if (dist_by_ml < dist_by_random) {
                ml_dist_win += 1
            }
            console.warn (this.color,'walk-step-ml', result);
            // this.main_color = 'green';
            classifierStep.addExample(action.toTensor_step(), _step);
        } else {
            random_dist_win += 1
            // this.main_color = 'red';
            console.error(this.color,'walk-step-Random', _step);
            classifierStep.addExample(action.toTensor_step(), _step);
        }

        g_data_dist.push([ loop_dist += 1, ml_dist_win, random_dist_win  ])

        return _step
    }

    async predictNav (food_min_dist, nav, pos) {
        let _nav = nav
        const food = food_min_dist

        const action = new Action({ x: food.x, y: food.y, nx: pos.x, ny: pos.y });
        const result = await classifier.predictClass(action.tensor);

        if (this.use_ai) {
            return result.label
        }

        const dist = (_nav_wasd) => {
            let x = pos.x
            let y = pos.y
            for (let index = 0; index < 2; index++) {
                const poin = this.nextPoint(_nav_wasd, x, y)
                x = poin.x
                y = poin.y
            }
            let dist = Math.pow(x - food.x, 2) + Math.pow( y - food.y, 2)
            dist = Math.sqrt(dist);
            return dist 
        }

        const dist_by_random =  dist(_nav);
        const dist_by_ml = dist(result.label);

        console.log(this.color, 'walk-dist_', { dist_by_ml, dist_by_random, label: result.label, _nav, nav: this.nav });

        if ((dist_by_ml) < (dist_by_random) &&  result.label != 'x') {
            if (dist_by_ml < dist_by_random) {
                ml_win += 1
            }
            this.main_color = 'blue';
            _nav = result.label;
            classifier.addExample(action.tensor, result.label);
            console.log(this.color,'walk-ml', result);
            console.log(this.color,'walk-ml', _nav);
        } else {
            this.main_color = 'red';
            random_win += 1
            console.log('walk-nav', action);
            console.log('walk-nav', { label:result.label, _nav, nav: this.nav});
            classifier.addExample(action.tensor, _nav);
            console.log(this.color,'walk-nav', result, {dist_by_ml, dist_by_random});
        }
        g_data.push([loop += 1, ml_win, random_win ])
        return _nav
    }

    nextPoint (nav, _x = null, _y = null) {
        let x = (_x == null) ? this.x : _x
        let y = (_y == null) ? this.y : _y
        const xfun = {
            w: () => y -= this.speed,
            s: () => y += this.speed,
            a: () => x -= this.speed,
            d: () => x += this.speed,
        }

        for (let index = 0; index < nav.length; index++) {
            const key = nav[index];
            if (xfun[key]) {
                xfun[key]()
            }
        }

        return { x, y }
    }

    walk() {
        console.log(this.color, 'walk', { step: this.step, nav:this.nav })
        if (this.step <= 0) { return; }
        const { x, y } = this.nextPoint(this.nav)
        this.y = y
        this.x = x
        
        this.step -= 1
    }

    eat (foods, callback) {
        const over = foods
            .filter(n => {
                const dist = Math.pow(this.x - n.x, 2) + Math.pow( this.y - n.y, 2)
                return Math.sqrt(dist) <= this.radius
            }) 

        if (over.length > 0) {
            over.forEach(food => {
                const nextFoods = foods.filter(xf => xf.id != food.id);
                food.x = parseInt(Math.random() * (this.canvas.width))
                food.y = parseInt(Math.random() * (this.canvas.height))
                nextFoods.push(food)
                if (this.target.id == food.id) {
                    this.target == null
                }
                
                eat_food += 1
                this.score += 1
                callback(nextFoods)
            })
        }
    }

    vision(nodes, callback) {
        const over = nodes
            .filter(n => n.id != this.id)
            .filter(n => {
                const dist = Math.pow(this.x - n.x, 2) + Math.pow( this.y - n.y, 2)
                return Math.sqrt(dist) <= this.radius
            }) 

        this.isDanger = over.length > 0
    }

    setColor() {
        this.color = this.main_color || 'orange'
        if (this.isDanger) {
            this.color =  'red' 
        }
    }
}

class Food {
    _x = 0
    _y = 0
    color = 'orange'
    radius = 2
    canvas = {
        height: 1,
        width: 1,
    }
    magin = 50

    constructor(w, h) {
        this.id = makeid(8)
        this.canvas = {
            width: w,
            height: h,
        }
        this.x = parseInt(Math.random() * (w))
        this.y = parseInt(Math.random() * (h))
    }

    get x() { return this._x }
    get y() { return this._y }

    set x(value) {
        if (value <= this.magin) {
            this._x =  this.magin
        } else if (value >= this.canvas.width - this.magin) {
            this._x = this.canvas.width - this.magin
        } else {
            this._x = value
        }
    }

    set y(value) {
        if (value <=  this.magin ) {
            this._y =  this.magin
        } else if (value >= this.canvas.height - this.magin) {
            this._y = this.canvas.height - this.magin
        } else {
            this._y = value
        }
    }
}

const distance = (xa, ya, xb, yb ) => {
    return Math.sqrt(Math.pow(xa - xb, 2) + Math.pow(ya - yb, 2))
}

window.onload = function() {
    // we'll put all our code within this function
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    const colors = ["blue", "green", "orange", "red", "#152c40"];
    let isMouseDown = false;
    document.getElementById('canvas-place').appendChild(canvas);
    canvas.width  = 800;
    canvas.height = 500
    // styles -------------------------
    document.body.style.background = "#000000";
    canvas.style.background = "#152c40";
    canvas.style.display = "block";
    canvas.style.margin = "0 auto";
    // --------------------------------

    const createChain = function(n, radius = 5) {
        const _nodes = [];
        
        for (let i = 0; i < n; i ++) {
          _nodes.push(new People(radius, canvas));
        }

        // const xxx = new People(radius, canvas);
        // xxx.use_ai = true
        // xxx.main_color = 'red'
        // _nodes.push(xxx)

        return _nodes;
    }

    const createFoods = function(n) {
        const _foods = [];
        console.log('createFoods', canvas.width)
        for (let i = 0; i < n; i ++) {
            _foods.push(new Food(canvas.width, canvas.height));
        }

        return _foods;
    }

    const update = function() {
        nodes.forEach(async node => {
            await node.think(foods)
            node.setColor()
            if (node.step > 0) {
                node.walk()
            }
            node.eat(foods, (next) => {foods = next})
            // node.vision(nodes, (next) => {nodes = next})

        })
      }
      const draw = function() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        nodes.forEach(node => {
          context.beginPath();
          context.fillStyle = node.color;
          context.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
          context.fillText(`score: ${node.score}`, node.x + node.radius + 2, node.y + 2);
          context.fillText(
            `${node.nav},${node.step}` ,
            node.x + node.radius + 2,
            node.y - 8
            );

          if (node.target) {
            const dist = (distance(node.x, node.y, node.target.x, node.target.y)).toFixed(4)
            if (dist > 200) {
                console.log({ "node.target": node.target })
            }
            context.fillText(
                `dist: ${dist}` ,
                node.x + node.radius + 2,
                node.y + 12
            );

            context.moveTo(node.x, node.y);
            context.strokeStyle = node.color;
            context.lineTo(node.target.x, node.target.y);
            context.stroke();
          }

          context.fill();
          context.closePath();
        })

        foods.forEach(food => {
            context.beginPath();
            context.fillStyle = food.color;
            context.arc(food.x, food.y, food.radius, 0, 2 * Math.PI);
            context.fill();
            context.closePath();
        })
      };

    let count = 0
    const start = new Date();
    const eLoop = document.getElementById('loop')
    const eDiffSec = document.getElementById('diff_sec')
    
    const tick = async function() {
        const b = new Date();
        const diff_sec = (b - start) / 1000;
        eLoop.innerText = count;
        eDiffSec.innerText = diff_sec;
        // eAiWork.innerText = addExample;
        // eEatFood.innerText = eat_food;
        // eXxx.innerText = eat_food / addExample;
       
        drawChart(g_data)
        drawChartDist(g_data_dist)
        drawChartMl(g_data_dist)

        update();
        draw();
        count += 1
        if (count % 100 == 0) {
            if ('x' in classifier.getClassExampleCount()) {
                classifier.clearClass('x')
            }
            classifier_save(classifier, 'classifier-nav');
            classifier_save(classifierStep, 'classifier-step');
            console.log('classifier-nav', {
                getClassExampleCount:  classifier.getClassExampleCount(),
                getClassifierDataset:  classifier.getClassifierDataset(),
                getNumClasses:  classifier.getNumClasses(),
            })
            console.log('classifier-step', {
                getClassExampleCount:  classifierStep.getClassExampleCount(),
                getClassifierDataset:  classifierStep.getClassifierDataset(),
                getNumClasses:  classifierStep.getNumClasses(),
            })


            if (count % 1000 == 0) {
                window.location.reload()
            }
        }
        requestAnimationFrame(tick);
    };


    canvas.onclick = function(e) {
        foods.push(new Food(e.offsetX, e.offsetY, canvas))
    };

    let foods = createFoods(200);
    let nodes = createChain(1); // you can also pass radius as a second param
    
    tick();

    document.getElementById('use-ai')
        .addEventListener('click', () => {
            nodes = nodes.map(n => {
                n.use_ai = !n.use_ai
                if (n.use_ai) {
                    n.main_color = 'blue'
                } else {
                    n.main_color = "red"
                }
                return n
            })
        })
}