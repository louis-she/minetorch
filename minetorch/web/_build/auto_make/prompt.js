'use strict'

var stepMap , cursor = 0, secure, stepCallback;
var userInput = '';

/**
 * 按照stepMap逐个执行
 * @param _stepMap  等待执行的step队列，JSON格式，格式如：
 * @param firstStep 从stepMap中的第几个开始执行，默认是第一个
 *
 * @example
 *  var prompt = require('prompt.js');
 *  prompt.startStepByStep({
 *      step1 : function(){},
 *      step2 : function(){}
 *  },0);
 */
var startStepByStep = function (_stepMap, firstStep) {
    stepMap = _stepMap;
    dataInputting();
    next(firstStep);
};

/**
 * 读取命令行的输入
 * @param tips      提示文字
 * @param callback  输入结束后的回调，格式为：function(data){}
 * @param secure    是否为安全码输入模式，默认：false
 *
 * @example
 *  var prompt = require('prompt.js');
 *  prompt.readLine('请输入密码：',function(data){
 *      var password = data;
 *  },true);
 */
var readLine = function (tips, callback, secure) {
    process.stdout.write(tips);
    setSecure(secure);
    stepCallback = callback;
};

/**
 * 获取当前处于执行中的Step
 * @return {*}
 */
var getCurrentStep = function () {
    var step_keys = Object.keys(stepMap);
    return stepMap[step_keys[cursor]];
};

/**
 * 执行下一个Step；可执行stepMap中指定位置的step
 * @param _cursor
 */
var next = function (_cursor) {
    cursor = +_cursor || cursor;
    var step = getCurrentStep();
    if (step) {
        step();
    } else {
        process.stdin.resume();
        process.stdin.end();
    }
};

/**
 * 设置是否为安全码输入模式，如果是，则在输入过程中，回显为“*”
 * @param _secure
 */
var setSecure = function (_secure) {
    secure = !!_secure;
    process.stdin.setRawMode(secure);
};

/**
 * 输入结束时的操作
 */
var dataFinished = function () {
    setSecure(false);
    userInput = userInput.toString().trim();
    var step = getCurrentStep();
    if (typeof step == 'function') {
        // 如果callback中返回true，则表示输入是合法的，可以进入下一步
        if (typeof stepCallback == 'function' && stepCallback(userInput)) {
            next(++cursor);
        } else {
            // 否则重复本步
            step();
        }
        userInput = '';
    } else {
        process.stdin.end();
    }
};

/**
 * 数据输入过程中的处理
 */
var dataInputting = function () {
    process.stdin.resume();
    process.stdin.setEncoding('utf8');
    process.stdin.setRawMode(false);

    /**
     * 监听数据输入，每次 Enter 则表示输入完毕
     */
    process.stdin.on("data", function (data) {
        // 如果是非安全码模式，直接回显，Enter后，结束操作
        if (!secure) {
            userInput = data;
            dataFinished();
            return;
        }

        // 安全码输入模式，回显为 *
        switch (data) {
            case "\n":  // 监听 Enter 键
            case "\r":
            case "\u0004":
                process.stdout.write('\n');
                dataFinished();
                break;
            case "\u0003": // Ctrl C
                process.exit();
                break;
            default:    // 其他字符回显为 *
                process.stdout.write('*');
                userInput += data;
                break;
        }
    });
};

exports.startStepByStep = startStepByStep;
exports.readLine = readLine;
