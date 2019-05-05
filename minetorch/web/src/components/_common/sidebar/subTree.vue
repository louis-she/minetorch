<template>
  <div>
    <template v-for="item in tree">
      <template v-if="item.subs">
        <el-submenu
          :key="item.id"
          :disabled="item.visible"
          :index="item.index">
          <template slot="title">
            <i :class="item.icon"></i>
            <span>{{ item.title }}</span>
          </template>
          <sub-tree :tree="item.subs"/>
        </el-submenu>
      </template>
      <template v-else>
        <el-menu-item
          :key="item.id"
          :disabled="item.visible == 0"
          :index="item.index">
          <i :class="item.icon"></i>{{ item.title }}
        </el-menu-item>
      </template>
    </template>
  </div>
</template>
<script>

export default {
  name: 'SubTree',
  props: {
    tree: {
      type: Array,
      default: () => []
    }
  }
}
</script>
<style lang="scss">
</style>
