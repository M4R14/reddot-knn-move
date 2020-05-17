import { Action } from "./Action";
import { classifier } from "../index";
import { makeid } from "../function/makeid";
import { Food } from "./Food";

interface Canvas {
    width: Number | bigint | undefined 
    height: Number | bigint | undefined 
}

function randomNumber(max: number, min: number = 0) : number {
    return Math.random() * (max - min) + min;
}

export class People {
    id: string;
    use_ai = false;
    main_color:string = '';
    _x = 0;
    nx = 0;
    ny = 0;
    _y = 0;
    step = 0;
    color = 'red';
    radius : number = 1;
    canvas: Canvas ;
    nav = 'w';
    speed = 1;
    score = 0;
    prediction : any = null;
    target: Food | undefined;

    constructor(radius: number, canvas: Canvas) {
        this.id = makeid(8);
        this.canvas = canvas;
        this.x = parseInt(randomNumber(Number(canvas.width) - radius).toString());
        this.y = parseInt(randomNumber(Number(canvas.height) - radius).toString());
        this.radius = radius;
    }
    
    get y() { return this._y; }
    set y(value) {
        const _height = Number(this.canvas.height)
        if (value - this.radius > 0 &&
            value + this.radius < _height) {
            this._y = value;
            return;
        }
        this._y = (value > _height / 2 ? _height - this.radius - 1 : this.radius + 1);
        return;
    }
    set x(value) {
        const _width = Number(this.canvas.height)
        if (value - this.radius > 0 &&
            value + this.radius < _width) {
            this._x = value;
            return;
        }
        this._x = (value > _width / 2 ? (_width - this.radius - 1) : this.radius + 1);
        return;
    }
    get x() { return this._x; }
    set_target(foods: any[]) {
        const pos = { x: this.x, y: this.y };
        let food_min_dist = foods
            .filter((f: { constructor: { name: string; }; }) => f.constructor.name == Food.name)
            .sort((a: { x: number; y: number; }, b: { x: number; y: number; }) => {
                const dist_a = Math.sqrt(Math.pow(pos.x - a.x, 2) + Math.pow(pos.y - a.y, 2));
                const dist_b = Math.sqrt(Math.pow(pos.x - b.x, 2) + Math.pow(pos.y - b.y, 2));
                return dist_a - dist_b;
            })[0];
        this.target = food_min_dist;
    }
    async think_v2() {
        const pos = { x: this.x, y: this.y };
        const control = [
            'w', 'a', 's', 'd',
            'wa', 'wd',
            'as', 'ds',
        ];
        // const steps = Array.from({ length: 10 }, (v, k) => k + 1);
        if (this.step <= 0) {
            let nextStep = 1;
            let nextNav = control[parseInt(randomNumber(control.length).toString())];
            let nextNav_arr = control;

            if (this.target) {
                const food_min_dist = this.target;
                nextNav = await this.predictNav_v2(food_min_dist, nextNav_arr, pos, nextStep);
            }
            this.nav = nextNav;
            this.step = nextStep;
        }
    }
    async predictNav_v2(food_min_dist: Food, nav: string[], pos: { x: any; y: any; }, nextStep: number) {
        let _nav = nav;
        const food = food_min_dist;
        const dist = (_nav_wasd: string | any[]) => {
            let x = pos.x;
            let y = pos.y;
            for (let index = 0; index < nextStep; index++) {
                const poin = this.nextPoint(_nav_wasd, x, y);
                x = poin.x;
                y = poin.y;
            }
            let dist = Math.pow(x - food.x, 2) + Math.pow(y - food.y, 2);
            dist = Math.sqrt(dist);
            return dist;
        };
        _nav = _nav.map((n: string) => [n, dist(n)])
        .reduce((a: any[], b: any[]) => {
            return a[1] > b[1] ? b : a;
        }, [0, Infinity])[0];
        const action = new Action({
            x: food.x,
            y: food.y,
            nx: pos.x,
            ny: pos.y,
            f: this.friends.map(f => {
                return Math.ceil(Math.sqrt(Math.pow(pos.x - f.x, 2) + Math.pow(pos.y - f.y, 2)));
            })
        });
        let result = null;
        try {
            result = await classifier.predictClass(action.tensor);
        }
        catch (error) {
            // console.log(error);
            classifier.addExample(action.tensor, _nav);
            return _nav;
        }
        if (this.use_ai) {
            return result.label;
        }
        const dist_by_random = dist(_nav);
        const dist_by_ml = dist(result.label);
        // console.log(this.color, 'walk-dist_', { dist_by_ml, dist_by_random, label: result.label, _nav, nav: this.nav });
        if ((dist_by_ml) <= (dist_by_random) && result.label != 'x') {
            if (dist_by_ml <= dist_by_random) {
                // ml_win += 1;
            }
            this.prediction = result;
            this.main_color = 'deepskyblue';
            _nav = result.label;
            // classifier.addExample(action.tensor, result.label);
            // console.log(this.color, 'walk-ml', result);
            // console.log(this.color, 'walk-ml', _nav);
        }
        else {
            this.prediction = null;
            this.main_color = 'red';
            // random_win += 1;
            // console.log('walk-nav', action);
            // console.log('walk-nav', { label: result.label, _nav, nav: this.nav });
            classifier.addExample(action.tensor, _nav);
            // console.log(this.color, 'walk-nav', result, { dist_by_ml, dist_by_random });
        }
        // g_data.push([loop += 1, ml_win, random_win]);
        return _nav;
    }

    friends: People[] = [];
    nextPoint(nav: string | any[], _x : number | null = null, _y  : number | null = null) {
        let _speed = this.speed;
        let x : number = (_x == null) ? this.x : _x;
        let y : number = (_y == null) ? this.y : _y;
        const xfun : any = {
            w: () => y -= _speed,
            s: () => y += _speed,
            a: () => x -= _speed,
            d: () => x += _speed,
        };
        for (let index = 0; index < nav.length; index++) {
            const key = nav[index];
            if (xfun[key]) {
                xfun[key]();
            }
        }

        this.friends.forEach((f: any) => {
            if (x >= f.x - f.radius && x <= f.x + f.radius) {
                if (this.score < f.score) {
                    x -= 1;
                }
            }
            if (y >= f.y - f.radius && y <= f.y + f.radius) {
                if (this.score < f.score) {
                    y -= 1;
                }
            }
        });
        return { x, y };
    }
    walk() {
        // console.log(this.color, 'walk', { step: this.step, nav: this.nav });
        if (this.step <= 0) {
            return;
        }
        const { x, y } = this.nextPoint(this.nav);
        this.y = y;
        this.x = x;
        this.step -= 1;
    }
    eat(foods: any[], callback: (arg0: any) => void) {
        const over = foods
            .filter(n => {
                const dist = Math.pow(this.x - n.x, 2) + Math.pow(this.y - n.y, 2);
                return Math.sqrt(dist) <= this.radius;
            });
        if (over.length > 0) {
            over.forEach(food => {
                const _width = Number(this.canvas.width)
                const _height = Number(this.canvas.height)
                const nextFoods = foods.filter(xf => xf.id != food.id);
                food.x = parseInt((Math.random() * _width).toString());
                food.y = parseInt((Math.random() * _height).toString());
                nextFoods.push(food);
                if (this.target) {
                    if (this.target.id == food.id) {
                        this.target == null;
                    }
                }
                // eat_food += 1;
                this.score += 1;
                callback(nextFoods);
            });
        }
    }
    vision(nodes: any[], callback: any) {
        const over = nodes
            .filter(n => n.id != this.id)
            .filter(n => {
                const dist = Math.pow(this.x - n.x, 2) + Math.pow(this.y - n.y, 2);
                return Math.sqrt(dist) <= this.radius;
            });
    }
    setColor() {
        this.color = this.main_color || 'orange';
    }
}
