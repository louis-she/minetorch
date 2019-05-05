'use strict'

const glob = require('glob');
const path = require('path');
const fs = require('fs');
require('shelljs/global');

const srcPath = path.resolve(__dirname, '../../src/templates') + '/';

const illegal = function(name) {
  return !/^[a-z0-9_]+?$/.test(name) || /^\d/.test(name)
}

const exist = function(name) {
  let srcModules = glob.sync(srcPath + '*').map(function(item) {
    return item.split('/').pop();
  });
  return srcModules.indexOf(name) > -1;
}


const _autoMakeIndexJs = function(name) {
  fs.writeFileSync(srcPath + name + '/index.js', `import template from './page.vue'
import info from './info'

export default {
  info,
  template
}

`);
};

const _autoMakeInfo = function(name) {
  fs.writeFileSync(srcPath + name + '/info.js', `export default {
  name: '${name}',
  title: 'xxx组件'
}

`);
};

const _autoMakePage = function(name) {
  fs.writeFileSync(srcPath + name + '/page.vue', `<template>
  <div class="component-container">
  </div>
</template>

<script>
import { langTrans } from 'utils/tools'

export default {
  props: {
    config: {
      type: Object,
      required: true
    },
    index: {
      type: Number,
      default () {
        return 0
      }
    },
    lang: {
      type: String,
      required: true
    }
  },
  data () {
    return {
      // model
      model: [
        {
          title: '模板一',
          data: {},
          schema: {
            title: 'xx组件',
            type: 'object',
            properties: {}
          }
        }
      ]
    }
  },
  // 初始化已有数据
  created () {
    this.model[this.index].data = Object.assign(this.model[this.index].data, this.config)
  },
  methods: {
    langTrans
  }
}
</script>

<style lang="scss" scpoed>
</style>

`);
};


const autoMake = function(name) {
  mkdir('-p', srcPath + name);
  _autoMakeIndexJs(name);
  _autoMakeInfo(name);
  _autoMakePage(name);
};

exports.illegal = illegal;
exports.exist = exist;
exports.autoMake = autoMake;
