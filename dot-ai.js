class Action {
    x = 0
    y = 0
    res= ''
    props = {}
    step = 999

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
        console.log(_q, _zeta, condition[_q] )
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
        return (Math.tan((y)/(x)))
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
            [(this.diff_x), (this.diff_y)],
            // [g(this.diff_x), g(this.diff_y)],
            // q([g(this.diff_x), g(this.diff_y)]),
            [this.x, this.y],
            // [this.nx, this.ny],
            // this.step,
            // l(this.dist),
            (this.dist),
            // this.zeta, 
        ]
       
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
            [(this.diff_x), (this.diff_y)],
            [l(this.diff_x), l(this.diff_y)],
            // this.res,
            q([g(this.diff_x), g(this.diff_y)]),
            // this.dist,
            l(this.dist),
            // this.zeta,
        ]
        console.log('walk-step toTensor_step', data)
        return tf.tensor(data)
    }
  }

data = [
    new Action({ y: 0, x: 0, res: 'w' }),
];

let addExample = 0
let eat_food = 0
let classifier  = null;

const init = async function() {
  // Create the classifier.
    classifier = knnClassifier.create();
    classifierStep = knnClassifier.create();


  // Add MobileNet activations to the model repeatedly for all classes.
  data.forEach(act => {
    classifier.addExample(act.tensor, act.res);
    classifierStep.addExample(act.toTensor_step(), act.step);
  })

  // Make a prediction.
  const x = new Action({ x: -5, y: 1 })
  const result = await classifier.predictClass(x.tensor);
  console.log(result);
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
    speed = 1
    _isDanger = false

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
        const control = [
            'w', 'a', 's', 'd',
            'wa', 'wd',
            'as', 'ds',
        ]

        if (this.step <= 0) {
            this.nav = control[parseInt(Math.random() * control.length)]
            let nextStep = parseInt(Math.random() * this.canvas.height)
            this.step = nextStep == 0 ? 1 : nextStep

            if (foods) {
                let food_min_dist = null
                const _foods = foods.filter(f => f.constructor.name == Food.name)
                    .map(n => {
                        const dist = Math.pow(this.x - n.x, 2) + Math.pow( this.y - n.y, 2)
                        n.dist = Math.sqrt(dist);
        
                        if (food_min_dist == null) {
                            food_min_dist = n
                        }  else {
                            if (food_min_dist.dist >  n.dist) {
                                food_min_dist = n
                            }
                        }
        
                        return n 
                    })
        
                if (food_min_dist) {
                    this.nav = await this.predictNav(food_min_dist, this.nav);
                    // this.step = Math.round(food_min_dist.dist)
                    const steps = [ 0, 1, 2, 4, 8, 16, 32, 50, 100, 200, 300 ]
                    const _steps = steps[parseInt(Math.random() * steps.length)]
                    // const _steps = parseInt(Math.random() * food_min_dist.dist)

                    this.step = await this.predictStep(food_min_dist, this.nav, _steps);
                    foods.filter(f => f.constructor.name == Food.name).map((f => {
                        f.color = "orange"
                        if (f.id == food_min_dist.id) {
                            f.color = 'green'
                        }
                    }))
                    // callback();
                }
            }
        }
    }

    async predictStep (food_min_dist, nav, step) {
        let _nav = nav
        let _step = Math.round(step)
        const food = food_min_dist
        // console.log('predictStep', {nav, step, _step});

        const action = new Action({ x: food.x, y: food.y, nx: this.x, ny: this.y, res: nav });
        let result = null
        try {
            result = await classifierStep.predictClass(action.toTensor_step());
        } catch (error) {
            classifierStep.addExample(action.toTensor_step(), Math.ceil(_step) );
            result = await classifierStep.predictClass(action.toTensor_step());
        }
       

        const dist_by_random = ((() => { 
           let x = this.x
           let y = this.y
            for (let index = 0; index < step; index++) {
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
            for (let index = 0; index < result.label; index++) {
                const poin = this.nextPoint(_nav, x, y)
                x = poin.x
                y = poin.y
            }

            let dist = Math.pow(x - food.x, 2) + Math.pow( y - food.y, 2)
            dist = Math.sqrt(dist);
            return dist
        })()); 

        if (dist_by_ml <= dist_by_random) {
            console.log('walk-step-ml', {
                _step,
                "label": result.label,
                'action.res': action.res,
                "action.dist": action.dist,
                // step,
                dist_by_ml,
                dist_by_random,
                result,
            });
            _step = result.label;
            
            ml_dist_win += 1
        } else {
            random_dist_win += 1
            console.log('walk-step', { _step: Math.ceil(_step), step }, action, {dist_by_ml, dist_by_random});
            classifierStep.addExample(action.toTensor_step(), Math.ceil(_step) );
        }

        g_data_dist.push([ loop_dist += 1, ml_dist_win, random_dist_win  ])

        return _step
    }

    async predictNav (food_min_dist, nav) {
        let _nav = nav
        const food = food_min_dist

        const action = new Action({ x: food.x, y: food.y, nx: this.x, ny: this.y });
        const result = await classifier.predictClass(action.tensor);

        const dist_by_random = ((() => { 
            const { x, y } = this.nextPoint(_nav)
            let dist = Math.pow(x - food.x, 2) + Math.pow( y - food.y, 2)
            dist = Math.sqrt(dist);
            return dist
        })()); 
        
        const dist_by_ml = ((() => { 
            const { x, y } = this.nextPoint(result.label)
            let dist = Math.pow(x - food.x, 2) + Math.pow( y - food.y, 2)
            dist = Math.sqrt(dist);
            return dist
        })()); 

        if (Math.round(dist_by_ml) <= Math.round(dist_by_random)) {
            _nav = result.label;
            ml_win += 1
            // console.log('walk-ml', _nav);
        } else {
            random_win += 1
            addExample += 1
            console.log('walk', action);
            classifier.addExample(action.tensor, _nav);
            console.log('walk', result, {dist_by_ml, dist_by_random});
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
                // food.x = parseInt(Math.random() * (this.canvas.width))
                // food.y = parseInt(Math.random() * (this.canvas.height))
                // nextFoods.push(food)
                eat_food += 1
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
        this.color = 'red'
        if (this.isDanger) {
            this.color = 'blue'
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

window.onload = function() {
    // we'll put all our code within this function
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    const colors = ["blue", "green", "orange", "red", "#2266FF"];
    let isMouseDown = false;
    document.getElementById('canvas-place').appendChild(canvas);
    canvas.width  = window.screen.width /2;
    canvas.height = window.screen.height /2
    // styles -------------------------
    document.body.style.background = "#000000";
    canvas.style.background = "#111111";
    canvas.style.display = "block";
    canvas.style.margin = "0 auto";
    // --------------------------------

    const createChain = function(n, radius = 5) {
        const _nodes = [];
        
        for (let i = 0; i < n; i ++) {
          _nodes.push(new People(radius, canvas));
        }
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
        nodes.forEach(node => {
            node.think(foods)
            node.walk()
            node.eat(foods, (next) => {foods = next})
            // node.vision(nodes, (next) => {nodes = next})
            node.setColor()
        })
      }
      const draw = function() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        nodes.forEach(node => {
          context.beginPath();
          context.fillStyle = node.color;
          context.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
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
    const eAiWork = document.getElementById('ai-work')
    const eEatFood = document.getElementById('eat-food')
    const eXxx = document.getElementById('xxx')
    
    const tick = function() {
        const b = new Date();
        const diff_sec = (b - start) / 1000;
        eLoop.innerText = count;
        eDiffSec.innerText = diff_sec;
        eAiWork.innerText = addExample;
        eEatFood.innerText = eat_food;
        eXxx.innerText = eat_food / addExample;
       
        drawChart(g_data)
        drawChartDist(g_data_dist)
        
        update();
        draw();
        requestAnimationFrame(tick);
        count += 1
    };


    canvas.onclick = function(e) {
        foods.push(new Food(e.offsetX, e.offsetY, canvas))
    };

    let foods = createFoods(500);
    let nodes = createChain(1); // you can also pass radius as a second param
    tick();
}