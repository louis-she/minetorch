'use strict';

const prompt = require('./prompt.js');
const tools = require('./tools.js');
let isExist = false;
let step1Text = '请给新建的组件命名：';

function _step1 () {
	prompt.readLine(step1Text, function (data) {
		let _name = `${data}-template`;
		if (tools.illegal(data)) {
			step1Text = '组件名只能包含小写字母、数字和下划线，并不能以数字开头，请重新输入新建的组件名：'
		} else if (tools.exist(_name)) {
			step1Text = data + '已存在，请重新输入新建的组件名：'
			isExist = true;
			return false;
		} else {
			step1Text = '';
			tools.autoMake(_name);
			return true;
		}
	})
}

prompt.startStepByStep({
	step1: _step1
})
