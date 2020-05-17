import { makeid } from "../function/makeid";

export class Food {
    id:string
    _x = 0;
    _y = 0;
    color = 'orange';
    radius = 2;
    canvas : { height: Number, width: Number };
    magin = 50;
    constructor(w: Number, h: Number) {
        this.id = makeid(8);
        this.canvas = {
            width: w,
            height: h,
        };
        this.x = parseInt((Math.random() * Number(w)).toString());
        this.y = parseInt((Math.random() * Number(h)).toString());
    }
    get x() { return this._x; }
    get y() { return this._y; }
    set x(value) {
        if (value <= this.magin) {
            this._x = this.magin;
        }
        else if (value >= Number(this.canvas.width) - this.magin) {
            this._x = Number(this.canvas.width) - this.magin;
        }
        else {
            this._x = value;
        }
    }
    set y(value) {
        if (value <= this.magin) {
            this._y = this.magin;
        }
        else if (value >= Number(this.canvas.height) - this.magin) {
            this._y = Number(this.canvas.height) - this.magin;
        }
        else {
            this._y = value;
        }
    }
}
